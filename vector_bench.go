package main

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"math"
	"math/rand"
	"net/url"
	"os"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"
	"unicode"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/credentials"
	"github.com/aws/aws-sdk-go/aws/session"
	"github.com/aws/aws-sdk-go/service/s3"
	gocb "github.com/couchbase/gocb/v2"
	"gopkg.in/ini.v1"
)

// ─────────────────────────────────────────────────────────────────────────────
// §1  Config
// ─────────────────────────────────────────────────────────────────────────────

type Config struct {
	// [connection]
	ConnectionString string
	Username         string
	Password         string
	Bucket           string
	Scope            string
	Collection       string

	// [dataset]
	QueryVectorsPath string
	GroundTruthPath  string
	DataPath         string
	ScalarFieldsPath string
	QueryFiltersPath string

	// [s3] (optional, used when dataset paths are s3:// URIs)
	S3Region          string
	S3Endpoint        string
	S3AccessKeyID     string
	S3SecretAccessKey string

	// [index]
	IndexType     string
	IndexName     string
	VectorField   string
	ScalarFields  []string
	IncludeFields []string
	IndexWhere    string

	// [index_creation]
	Dimension         int
	Similarity        string
	Description       string
	NList             int
	TrainList         int
	NumReplica        int
	PersistFullVector bool

	// [query]
	TopK           int
	NProbes        int
	Reranking      bool
	TopNScan       int
	WhereClause    string
	WhereClauseSQL string

	// [benchmark]
	Workers      int
	Warmup       int
	Duration     int
	QueryTimeout time.Duration

	// [loading]
	SkipLoad    bool
	SkipIndex   bool
	BatchSize   int
	LoadWorkers int
	DocIDPrefix string

	// [output]
	OutputJSON string
}

func loadConfig(path string) (*Config, error) {
	f, err := ini.Load(path)
	if err != nil {
		return nil, fmt.Errorf("cannot read config %s: %w", path, err)
	}

	c := &Config{}

	conn := f.Section("connection")
	c.ConnectionString = conn.Key("connection_string").String()
	c.Username = conn.Key("username").String()
	c.Password = conn.Key("password").String()
	c.Bucket = conn.Key("bucket").MustString("vector-bench")
	c.Scope = conn.Key("scope").MustString("_default")
	c.Collection = conn.Key("collection").MustString("_default")

	ds := f.Section("dataset")
	c.QueryVectorsPath = ds.Key("query_vectors_path").String()
	c.GroundTruthPath = ds.Key("ground_truth_path").String()
	c.DataPath = ds.Key("data_path").String()
	c.ScalarFieldsPath = ds.Key("scalar_fields_path").MustString("")
	c.QueryFiltersPath = ds.Key("query_filters_path").MustString("")

	s3 := f.Section("s3")
	c.S3Region = s3.Key("region").MustString("")
	c.S3Endpoint = s3.Key("endpoint").MustString("")
	c.S3AccessKeyID = s3.Key("access_key_id").MustString("")
	c.S3SecretAccessKey = s3.Key("secret_access_key").MustString("")

	idx := f.Section("index")
	c.IndexType = normalizeIndexType(idx.Key("index_type").MustString("hyperscale"))
	c.IndexName = idx.Key("index_name").MustString("vec_bench_idx")
	c.VectorField = idx.Key("vector_field").MustString("vec")
	c.ScalarFields = parseCSVList(idx.Key("scalar_fields").MustString(""))
	c.IncludeFields = parseCSVList(idx.Key("include_fields").MustString(""))
	c.IndexWhere = idx.Key("where_clause").MustString("")

	ic := f.Section("index_creation")
	c.Dimension = ic.Key("dimension").MustInt(128)
	c.Similarity = ic.Key("similarity").MustString("L2")
	c.Description = ic.Key("description").MustString("IVF,SQ8")
	c.NList = ic.Key("nlist").MustInt(1024)
	c.TrainList = ic.Key("train_list").MustInt(100000)
	c.NumReplica = ic.Key("num_replica").MustInt(0)
	c.PersistFullVector = ic.Key("persist_full_vector").MustBool(true)

	q := f.Section("query")
	c.TopK = q.Key("top_k").MustInt(10)
	c.NProbes = q.Key("nprobes").MustInt(1)
	c.Reranking = q.Key("reranking").MustBool(false)
	c.TopNScan = q.Key("top_n_scan").MustInt(0)
	c.WhereClause = q.Key("where_clause").MustString("")
	if strings.TrimSpace(c.WhereClause) != "" {
		whereSQL, err := parseWhereClauseJSON(c.WhereClause)
		if err != nil {
			return nil, fmt.Errorf("invalid [query].where_clause JSON: %w", err)
		}
		c.WhereClauseSQL = whereSQL
	}

	b := f.Section("benchmark")
	c.Workers = b.Key("workers").MustInt(16)
	c.Warmup = b.Key("warmup_seconds").MustInt(30)
	c.Duration = b.Key("steady_state_seconds").MustInt(120)
	c.QueryTimeout = time.Duration(b.Key("query_timeout_ms").MustInt(10000)) * time.Millisecond

	l := f.Section("loading")
	c.SkipLoad = l.Key("skip_load").MustBool(false)
	c.SkipIndex = l.Key("skip_index").MustBool(false)
	c.BatchSize = l.Key("load_batch_size").MustInt(500)
	c.LoadWorkers = l.Key("load_workers").MustInt(c.Workers)
	c.DocIDPrefix = l.Key("doc_id_prefix").MustString("vec")

	o := f.Section("output")
	c.OutputJSON = o.Key("output_json").MustString("results.json")

	return c, nil
}

func parseJSONMap(raw string) (map[string]interface{}, error) {
	trimmed := strings.TrimSpace(raw)
	if trimmed == "" {
		return map[string]interface{}{}, nil
	}
	out := map[string]interface{}{}
	if err := json.Unmarshal([]byte(trimmed), &out); err != nil {
		return nil, err
	}
	return out, nil
}

func parseWhereClauseJSON(raw string) (string, error) {
	var payload interface{}
	if err := json.Unmarshal([]byte(raw), &payload); err != nil {
		return "", err
	}
	sql, err := buildWhereExpr(payload)
	if err != nil {
		return "", err
	}
	return sql, nil
}

func loadJSONObjects(path string, opener *datasetOpener, label string) ([]map[string]interface{}, error) {
	if strings.TrimSpace(path) == "" {
		return nil, nil
	}
	rc, err := opener.Open(path)
	if err != nil {
		return nil, fmt.Errorf("failed to open %s at %s: %w", label, path, err)
	}
	defer rc.Close()

	var out []map[string]interface{}
	if err := json.NewDecoder(rc).Decode(&out); err != nil {
		return nil, fmt.Errorf("failed to decode %s JSON array at %s: %w", label, path, err)
	}
	return out, nil
}

func buildWhereExpr(v interface{}) (string, error) {
	obj, ok := v.(map[string]interface{})
	if !ok {
		return "", fmt.Errorf("where clause must be a JSON object")
	}

	if andRaw, exists := obj["and"]; exists {
		items, ok := andRaw.([]interface{})
		if !ok || len(items) == 0 {
			return "", fmt.Errorf("\"and\" must be a non-empty array")
		}
		parts := make([]string, 0, len(items))
		for _, it := range items {
			part, err := buildWhereExpr(it)
			if err != nil {
				return "", err
			}
			parts = append(parts, part)
		}
		return "(" + strings.Join(parts, " AND ") + ")", nil
	}

	if orRaw, exists := obj["or"]; exists {
		items, ok := orRaw.([]interface{})
		if !ok || len(items) == 0 {
			return "", fmt.Errorf("\"or\" must be a non-empty array")
		}
		parts := make([]string, 0, len(items))
		for _, it := range items {
			part, err := buildWhereExpr(it)
			if err != nil {
				return "", err
			}
			parts = append(parts, part)
		}
		return "(" + strings.Join(parts, " OR ") + ")", nil
	}

	parts := make([]string, 0, len(obj))
	for k, val := range obj {
		parts = append(parts, fmt.Sprintf("b.%s = %s", formatFieldExpr(k), formatSQLLiteral(val)))
	}
	if len(parts) == 0 {
		return "", nil
	}
	if len(parts) == 1 {
		return parts[0], nil
	}
	return "(" + strings.Join(parts, " AND ") + ")", nil
}

func formatSQLLiteral(v interface{}) string {
	switch x := v.(type) {
	case nil:
		return "NULL"
	case string:
		return "'" + strings.ReplaceAll(x, "'", "''") + "'"
	case bool:
		if x {
			return "TRUE"
		}
		return "FALSE"
	case float64:
		return fmt.Sprintf("%v", x)
	default:
		b, _ := json.Marshal(x)
		return "'" + strings.ReplaceAll(string(b), "'", "''") + "'"
	}
}

func parseCSVList(raw string) []string {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return nil
	}
	parts := strings.Split(raw, ",")
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p != "" {
			out = append(out, p)
		}
	}
	return out
}

func normalizeIndexType(raw string) string {
	return strings.ToLower(strings.TrimSpace(raw))
}

func isHyperscaleIndexType(indexType string) bool {
	return normalizeIndexType(indexType) == "hyperscale"
}

func isCompositeIndexType(indexType string) bool {
	return normalizeIndexType(indexType) == "composite"
}

// ─────────────────────────────────────────────────────────────────────────────
// §2  CLI flag overrides
//
//	All flags default to values from config.ini.
//	Flags passed on the command line override the INI value.
//
// ─────────────────────────────────────────────────────────────────────────────
func parseFlags(c *Config) bool {
	// Register -config so flag.Parse() doesn't reject it.
	// The actual value was already read in main() before INI was loaded.
	flag.String("config", "config.ini", "Path to config.ini")

	nlist := flag.Int("nlist", c.NList, "Number of IVF centroids")
	trainList := flag.Int("train-list", c.TrainList, "Training set size for codebook")
	quantization := flag.String("quantization", "", "Quantization suffix e.g. SQ8, SQ4")
	persistFV := flag.String("persist-full-vector", "", "Store full vector: true|false")
	nprobes := flag.Int("nprobes", c.NProbes, "nProbes per query")
	reranking := flag.String("reranking", "", "Enable reranking: true|false")
	topNScan := flag.Int("top-n-scan", c.TopNScan, "topNScan (0 = default)")
	workers := flag.Int("workers", c.Workers, "Concurrent query workers")
	warmup := flag.Int("warmup", c.Warmup, "Warmup duration in seconds")
	duration := flag.Int("duration", c.Duration, "Measurement duration in seconds")
	topK := flag.Int("top-k", c.TopK, "K for KNN retrieval")
	skipLoad := flag.String("skip-load", "", "Skip data load: true|false")
	skipIndex := flag.String("skip-index", "", "Skip index build: true|false")
	loadWorkers := flag.Int("load-workers", c.LoadWorkers, "Concurrent data load workers")
	output := flag.String("output", c.OutputJSON, "Output JSON path")
	dropIndexOnly := flag.Bool("drop-index-only", false, "Drop configured index and exit")
	flag.Parse()

	c.NList = *nlist
	c.TrainList = *trainList
	c.NProbes = *nprobes
	c.TopNScan = *topNScan
	c.Workers = *workers
	c.Warmup = *warmup
	c.Duration = *duration
	c.TopK = *topK
	c.LoadWorkers = *loadWorkers
	c.OutputJSON = *output

	// Build effective description: IVF<nlist>,<quantization>
	if *quantization != "" {
		c.Description = fmt.Sprintf("IVF%d,%s", c.NList, *quantization)
	} else {
		// Substitute nlist into existing description
		parts := strings.SplitN(c.Description, ",", 2)
		if len(parts) == 2 && c.NList > 0 {
			c.Description = fmt.Sprintf("IVF%d,%s", c.NList, parts[1])
		}
	}

	if *persistFV != "" {
		c.PersistFullVector = strings.ToLower(*persistFV) == "true"
	}
	if *reranking != "" {
		c.Reranking = strings.ToLower(*reranking) == "true"
	}
	// Reranking requires full-vector persistence, but full-vector persistence
	// does not force reranking on.
	if c.Reranking && !c.PersistFullVector {
		log.Printf("reranking requested but persist_full_vector=false; forcing reranking=false")
		c.Reranking = false
	}
	if !c.Reranking {
		c.TopNScan = 0
	}
	if *skipLoad != "" {
		c.SkipLoad = strings.ToLower(*skipLoad) == "true"
	}
	if *skipIndex != "" {
		c.SkipIndex = strings.ToLower(*skipIndex) == "true"
	}
	return *dropIndexOnly
}

// ─────────────────────────────────────────────────────────────────────────────
// §3  Dataset source abstraction + .fvecs / .ivecs readers
// ─────────────────────────────────────────────────────────────────────────────

type datasetOpener struct {
	cfg      *Config
	mu       sync.Mutex
	s3Client *s3.S3
}

func newDatasetOpener(c *Config) *datasetOpener {
	return &datasetOpener{cfg: c}
}

func isS3Path(path string) bool {
	return strings.HasPrefix(strings.ToLower(path), "s3://")
}

func parseS3URI(raw string) (bucket, key string, err error) {
	u, err := url.Parse(raw)
	if err != nil {
		return "", "", fmt.Errorf("invalid S3 URI %q: %w", raw, err)
	}
	if u.Scheme != "s3" {
		return "", "", fmt.Errorf("invalid S3 URI scheme for %q", raw)
	}
	if u.Host == "" {
		return "", "", fmt.Errorf("missing bucket in S3 URI %q", raw)
	}
	key = strings.TrimPrefix(u.Path, "/")
	if key == "" {
		return "", "", fmt.Errorf("missing object key in S3 URI %q", raw)
	}
	return u.Host, key, nil
}

func (o *datasetOpener) getS3Client() (*s3.S3, error) {
	o.mu.Lock()
	defer o.mu.Unlock()
	if o.s3Client != nil {
		return o.s3Client, nil
	}

	region := o.cfg.S3Region
	if region == "" {
		region = "us-east-1"
	}

	awsCfg := aws.NewConfig().
		WithRegion(region)
	if o.cfg.S3Endpoint != "" {
		awsCfg = awsCfg.WithEndpoint(o.cfg.S3Endpoint)
	}
	if o.cfg.S3AccessKeyID != "" && o.cfg.S3SecretAccessKey != "" {
		awsCfg = awsCfg.WithCredentials(credentials.NewStaticCredentials(
			o.cfg.S3AccessKeyID,
			o.cfg.S3SecretAccessKey,
			"",
		))
	}

	sessOpts := session.Options{
		Config:            *awsCfg,
		SharedConfigState: session.SharedConfigEnable,
	}
	sess, err := session.NewSessionWithOptions(sessOpts)
	if err != nil {
		return nil, fmt.Errorf("failed to create AWS session: %w", err)
	}

	client := s3.New(sess, awsCfg)

	o.s3Client = client
	return client, nil
}

func (o *datasetOpener) Open(path string) (io.ReadCloser, error) {
	if !isS3Path(path) {
		return os.Open(path)
	}

	bucket, key, err := parseS3URI(path)
	if err != nil {
		return nil, err
	}
	client, err := o.getS3Client()
	if err != nil {
		return nil, err
	}
	out, err := client.GetObject(&s3.GetObjectInput{
		Bucket: aws.String(bucket),
		Key:    aws.String(key),
	})
	if err != nil {
		return nil, fmt.Errorf("s3 get object %s: %w", path, err)
	}
	return out.Body, nil
}

func readFvec(r io.Reader) ([]float32, error) {
	var dim int32
	if err := binary.Read(r, binary.LittleEndian, &dim); err != nil {
		return nil, err
	}
	if dim <= 0 {
		return nil, fmt.Errorf("invalid fvec dimension: %d", dim)
	}
	vec := make([]float32, int(dim))
	if err := binary.Read(r, binary.LittleEndian, vec); err != nil {
		return nil, err
	}
	return vec, nil
}

func readIvec(r io.Reader) ([]int32, error) {
	var dim int32
	if err := binary.Read(r, binary.LittleEndian, &dim); err != nil {
		return nil, err
	}
	if dim <= 0 {
		return nil, fmt.Errorf("invalid ivec dimension: %d", dim)
	}
	vec := make([]int32, int(dim))
	if err := binary.Read(r, binary.LittleEndian, vec); err != nil {
		return nil, err
	}
	return vec, nil
}

func streamFvecs(path string, opener *datasetOpener, emit func(idx int, vec []float32) error) (int, error) {
	r, err := opener.Open(path)
	if err != nil {
		return 0, err
	}
	defer r.Close()

	count := 0
	for {
		vec, err := readFvec(r)
		if err != nil {
			if errors.Is(err, io.EOF) {
				break
			}
			return count, err
		}
		if err := emit(count, vec); err != nil {
			return count, err
		}
		count++
	}
	return count, nil
}

func loadFvecs(path string, opener *datasetOpener) ([][]float32, error) {
	r, err := opener.Open(path)
	if err != nil {
		return nil, err
	}
	defer r.Close()

	var vecs [][]float32
	for {
		vec, err := readFvec(r)
		if err != nil {
			if errors.Is(err, io.EOF) {
				break
			}
			return nil, err
		}
		vecs = append(vecs, vec)
	}
	return vecs, nil
}

func loadIvecs(path string, opener *datasetOpener) ([][]int32, error) {
	r, err := opener.Open(path)
	if err != nil {
		return nil, err
	}
	defer r.Close()

	var vecs [][]int32
	for {
		vec, err := readIvec(r)
		if err != nil {
			if errors.Is(err, io.EOF) {
				break
			}
			return nil, err
		}
		vecs = append(vecs, vec)
	}
	return vecs, nil
}

// ─────────────────────────────────────────────────────────────────────────────
// §4  Couchbase connection
// ─────────────────────────────────────────────────────────────────────────────

func connectCB(c *Config) (*gocb.Cluster, error) {
	log.Printf("Connecting to %s ...", c.ConnectionString)
	cluster, err := gocb.Connect(c.ConnectionString, gocb.ClusterOptions{
		Username: c.Username,
		Password: c.Password,
	})
	if err != nil {
		return nil, fmt.Errorf("connect: %w", err)
	}
	if err := cluster.WaitUntilReady(30*time.Second, nil); err != nil {
		return nil, fmt.Errorf("wait ready: %w", err)
	}
	log.Printf("Connected.")
	return cluster, nil
}

// ─────────────────────────────────────────────────────────────────────────────
// §5  Management query helper
// ─────────────────────────────────────────────────────────────────────────────

func mgmtQuery(cluster *gocb.Cluster, stmt string, timeout time.Duration) ([]map[string]interface{}, error) {
	result, err := cluster.Query(stmt, &gocb.QueryOptions{
		Adhoc:   true,
		Timeout: timeout,
	})
	if err != nil {
		return nil, err
	}
	var rows []map[string]interface{}
	for result.Next() {
		var row map[string]interface{}
		if err := result.Row(&row); err != nil {
			continue
		}
		rows = append(rows, row)
	}
	return rows, result.Err()
}

// ─────────────────────────────────────────────────────────────────────────────
// §6  Index creation + polling
// ─────────────────────────────────────────────────────────────────────────────

func createIndex(cluster *gocb.Cluster, c *Config) (float64, error) {
	ks := fmt.Sprintf("`%s`.`%s`.`%s`", c.Bucket, c.Scope, c.Collection)

	// Drop existing
	log.Printf("Dropping existing index '%s' if present ...", c.IndexName)
	mgmtQuery(cluster, fmt.Sprintf("DROP INDEX `%s` IF EXISTS ON %s;", c.IndexName, ks), 30*time.Second)

	// Build WITH clause
	with := fmt.Sprintf(
		`{"dimension":%d,"similarity":"%s","description":"%s","train_list":%d,"persist_full_vector":%t}`,
		c.Dimension, c.Similarity, c.Description, c.TrainList, c.PersistFullVector,
	)
	vectorKey := fmt.Sprintf("%s VECTOR", formatFieldExpr(c.VectorField))
	partitionKeys := formatFieldExprList(c.ScalarFields)
	includeKeys := formatFieldExprList(c.IncludeFields)
	whereClause := strings.TrimSpace(c.IndexWhere)
	indexType := normalizeIndexType(c.IndexType)
	isHyperscale := false

	// Build DDL
	var ddl string
	switch {
	case isHyperscaleIndexType(indexType):
		isHyperscale = true
		keys := append([]string{}, partitionKeys...)
		keys = append(keys, vectorKey)
		ddl = fmt.Sprintf("CREATE VECTOR INDEX `%s`\n  ON %s (%s)", c.IndexName, ks, strings.Join(keys, ", "))
	case isCompositeIndexType(indexType):
		compositeKeys := append([]string{}, partitionKeys...)
		compositeKeys = append(compositeKeys, vectorKey)
		ddl = fmt.Sprintf("CREATE INDEX `%s`\n  ON %s (%s)\n  ", c.IndexName, ks, strings.Join(compositeKeys, ", "))
	default:
		return 0, fmt.Errorf("unsupported index_type %q (supported: hyperscale, composite)", c.IndexType)
	}
	if len(includeKeys) > 0 && isHyperscale {
		ddl += fmt.Sprintf("\n  INCLUDE (%s)", strings.Join(includeKeys, ", "))
	} else if len(includeKeys) > 0 {
		log.Printf("Ignoring include_fields for index_type=%s (INCLUDE is hyperscale-only).", c.IndexType)
	}
	if whereClause != "" {
		ddl += fmt.Sprintf("\n  WHERE %s", whereClause)
	}

	ddl += fmt.Sprintf("\n  WITH %s", with)
	log.Printf("Creating index:\n%s", ddl)

	t0 := time.Now()
	if _, err := mgmtQuery(cluster, ddl, 10*time.Minute); err != nil {
		// AmbiguousTimeoutException is common — index still builds in background
		log.Printf("  CREATE returned: %v (index may still build in background)", err)
	}

	// Poll system:indexes until state = online
	// keyspace_id holds the bucket name in Couchbase's system catalog
	pollStmt := fmt.Sprintf(
		"SELECT state, name FROM system:indexes WHERE name = '%s' AND keyspace_id = '%s'",
		c.IndexName, c.Bucket,
	)
	deadline := time.Now().Add(60 * time.Minute)
	log.Printf("Polling for index state = online ...")
	lastState := ""
	for time.Now().Before(deadline) {
		rows, err := mgmtQuery(cluster, pollStmt, 30*time.Second)
		if err != nil {
			log.Printf("  Poll error: %v", err)
		} else if len(rows) == 0 {
			log.Printf("  Index not yet visible in system:indexes ...")
		} else {
			state := fmt.Sprintf("%v", rows[0]["state"])
			if state != lastState {
				log.Printf("  state: %s", state)
				lastState = state
			}
			if state == "online" {
				log.Printf("  ✓ Index is online!")
				break
			}
		}
		time.Sleep(10 * time.Second)
	}

	buildTime := time.Since(t0).Seconds()
	log.Printf("Index build time: %.1fs", buildTime)
	return buildTime, nil
}

func dropIndex(cluster *gocb.Cluster, c *Config) error {
	ks := fmt.Sprintf("`%s`.`%s`.`%s`", c.Bucket, c.Scope, c.Collection)
	log.Printf("Dropping index '%s' ...", c.IndexName)
	_, err := mgmtQuery(cluster, fmt.Sprintf("DROP INDEX `%s` IF EXISTS ON %s;", c.IndexName, ks), 60*time.Second)
	return err
}

func isSimpleIdentifier(field string) bool {
	if field == "" {
		return false
	}
	for i, r := range field {
		if i == 0 {
			if !(r == '_' || unicode.IsLetter(r)) {
				return false
			}
		} else if !(r == '_' || unicode.IsLetter(r) || unicode.IsDigit(r)) {
			return false
		}
	}
	return true
}

func formatFieldExpr(field string) string {
	field = strings.TrimSpace(field)
	if field == "" {
		return ""
	}
	if isSimpleIdentifier(field) {
		return fmt.Sprintf("`%s`", field)
	}
	return field
}

func buildIndexKeys(c *Config) string {
	parts := []string{fmt.Sprintf("%s VECTOR", formatFieldExpr(c.VectorField))}
	for _, scalar := range c.ScalarFields {
		expr := formatFieldExpr(scalar)
		if expr != "" {
			parts = append(parts, expr)
		}
	}
	return strings.Join(parts, ", ")
}

func formatFieldExprList(fields []string) []string {
	out := make([]string, 0, len(fields))
	for _, field := range fields {
		expr := formatFieldExpr(field)
		if expr != "" {
			out = append(out, expr)
		}
	}
	return out
}

// ─────────────────────────────────────────────────────────────────────────────
// §7  Data loading  — Go goroutines for maximum parallel throughput
//
//	Each document is a {"vec": [float32...]} JSON object.
//	A worker pool batches upserts using Collection.Do for higher throughput.
//
// ─────────────────────────────────────────────────────────────────────────────

func loadData(cluster *gocb.Cluster, c *Config, opener *datasetOpener, scalarRows []map[string]interface{}) error {
	col := cluster.Bucket(c.Bucket).Scope(c.Scope).Collection(c.Collection)
	type docItem struct {
		id  string
		vec []float32
		row map[string]interface{}
	}

	loadWorkers := c.LoadWorkers
	if loadWorkers <= 0 {
		loadWorkers = c.Workers
	}
	if loadWorkers <= 0 {
		loadWorkers = 1
	}
	batchSize := c.BatchSize
	if batchSize <= 0 {
		batchSize = 1
	}

	log.Printf("Loading vectors from %s (%d workers, batch=%d) ...", c.DataPath, loadWorkers, batchSize)
	t0 := time.Now()

	jobs := make(chan docItem, loadWorkers*batchSize)
	var wg sync.WaitGroup
	var successCount, errCount int64

	worker := func() {
		defer wg.Done()
		ops := make([]gocb.BulkOp, 0, batchSize)
		flush := func() {
			if len(ops) == 0 {
				return
			}
			if err := col.Do(ops, &gocb.BulkOpOptions{
				Timeout: 30 * time.Second,
			}); err != nil {
				log.Printf("  Bulk upsert error: %v", err)
			}
			for _, op := range ops {
				if up, ok := op.(*gocb.UpsertOp); ok && up.Err != nil {
					atomic.AddInt64(&errCount, 1)
				} else {
					atomic.AddInt64(&successCount, 1)
				}
			}
			ops = ops[:0]
		}

		for item := range jobs {
			doc := map[string]interface{}{c.VectorField: item.vec}
			for k, v := range item.row {
				doc[k] = v
			}
			ops = append(ops, &gocb.UpsertOp{ID: item.id, Value: doc})
			if len(ops) >= batchSize {
				flush()
			}
		}
		flush()
	}

	for i := 0; i < loadWorkers; i++ {
		wg.Add(1)
		go worker()
	}

	enqueued, streamErr := streamFvecs(c.DataPath, opener, func(i int, vec []float32) error {
		row := map[string]interface{}{}
		if len(scalarRows) > 0 {
			if i >= len(scalarRows) {
				return fmt.Errorf("scalar_fields_path has %d rows but dataset has more vectors (first missing row index=%d)", len(scalarRows), i)
			}
			row = scalarRows[i]
		}
		jobs <- docItem{
			id:  fmt.Sprintf("%s-%d", c.DocIDPrefix, i),
			vec: vec,
			row: row,
		}
		if (i+1)%10000 == 0 {
			log.Printf("  Enqueued %d documents ...", i+1)
		}
		return nil
	})
	close(jobs)
	wg.Wait()
	if streamErr != nil {
		return fmt.Errorf("failed while streaming base vectors: %w", streamErr)
	}

	log.Printf("Data load complete: %d enqueued, %d ok, %d errors in %.1fs",
		enqueued,
		atomic.LoadInt64(&successCount),
		atomic.LoadInt64(&errCount),
		time.Since(t0).Seconds(),
	)
	return nil
}

// ─────────────────────────────────────────────────────────────────────────────
// §8  Query building
// ─────────────────────────────────────────────────────────────────────────────

func combineWhereClauses(a, b string) string {
	a = strings.TrimSpace(a)
	b = strings.TrimSpace(b)
	if a == "" {
		return b
	}
	if b == "" {
		return a
	}
	return fmt.Sprintf("(%s) AND (%s)", a, b)
}

func buildQueryStmt(c *Config, whereSQL string) string {
	ks := fmt.Sprintf("`%s`.`%s`.`%s`", c.Bucket, c.Scope, c.Collection)

	// Build APPROX_VECTOR_DISTANCE arguments
	avd := fmt.Sprintf(`"%s", %d`, c.Similarity, c.NProbes) // similarity, nProbes
	if c.Reranking {
		avd += ", TRUE"
	} else {
		avd += ", FALSE"
	}

	if isHyperscaleIndexType(c.IndexType) && c.TopNScan > 0 {
		avd += fmt.Sprintf(", %d", c.TopNScan) // topNScan is hyperscale-only
	}

	stmt := fmt.Sprintf(
		"SELECT META(b).id FROM %s AS b",
		ks,
	)
	if strings.TrimSpace(whereSQL) != "" {
		stmt += " WHERE " + whereSQL
	}
	stmt += fmt.Sprintf(
		" ORDER BY APPROX_VECTOR_DISTANCE(%s, $vec, %s) LIMIT %d",
		fmt.Sprintf("b.%s", formatFieldExpr(c.VectorField)),
		avd,
		c.TopK,
	)
	return stmt
}

// ─────────────────────────────────────────────────────────────────────────────
// §9  Benchmark engine  — closed-loop parallel query execution
//
//	Architecture:
//	  - c.Workers goroutines run in parallel (closed loop)
//	  - Each goroutine continuously picks a random query vector,
//	    fires the query, and records the result
//	  - All workers stop after `durationSec` seconds
//	  - Results are collected for metric computation
//
// ─────────────────────────────────────────────────────────────────────────────

type queryResult struct {
	queryIdx int           // index into queryVecs (used for recall computation)
	ids      []string      // returned document IDs
	latency  time.Duration // end-to-end query latency
}

func runPhase(cluster *gocb.Cluster, c *Config, queryStmts []string, queryVecs [][]float32, durationSec int) []queryResult {
	var (
		results []queryResult
		mu      sync.Mutex
		wg      sync.WaitGroup
		stop    int32 // atomic flag: 1 = stop all workers
	)

	n := len(queryVecs)

	// Each worker has its own RNG to avoid lock contention
	worker := func(seed int64) {
		defer wg.Done()
		rng := rand.New(rand.NewSource(seed))

		for atomic.LoadInt32(&stop) == 0 {
			qIdx := rng.Intn(n)
			stmt := queryStmts[0]
			if len(queryStmts) == n {
				stmt = queryStmts[qIdx]
			}

			start := time.Now()
			rows, err := cluster.Query(stmt, &gocb.QueryOptions{
				Adhoc:           true,
				Timeout:         c.QueryTimeout,
				NamedParameters: map[string]interface{}{"vec": queryVecs[qIdx]},
			})
			lat := time.Since(start)

			if err != nil {
				continue // count as failed; don't record
			}

			var ids []string
			for rows.Next() {
				var row struct {
					ID string `json:"id"`
				}
				if err := rows.Row(&row); err == nil {
					ids = append(ids, row.ID)
				}
			}
			rows.Close()

			mu.Lock()
			results = append(results, queryResult{queryIdx: qIdx, ids: ids, latency: lat})
			mu.Unlock()
		}
	}

	// Launch all workers
	for i := 0; i < c.Workers; i++ {
		wg.Add(1)
		go worker(int64(i + 1))
	}

	// Let them run for durationSec, then signal stop
	time.Sleep(time.Duration(durationSec) * time.Second)
	atomic.StoreInt32(&stop, 1)
	wg.Wait()

	return results
}

// ─────────────────────────────────────────────────────────────────────────────
// §10  Metrics computation
// ─────────────────────────────────────────────────────────────────────────────

type Metrics struct {
	TotalQueries      int     `json:"total_queries"`
	SuccessfulQueries int     `json:"successful_queries"`
	FailedQueries     int     `json:"failed_queries"`
	ThroughputQPS     float64 `json:"throughput_qps"`
	LatencyAvgMs      float64 `json:"latency_avg_ms"`
	LatencyP95Ms      float64 `json:"latency_p95_ms"`
	LatencyP99Ms      float64 `json:"latency_p99_ms"`
	RecallAt10        float64 `json:"recall_at_10"`
	IndexBuildTimeS   float64 `json:"index_build_time_s,omitempty"`
}

func computeMetrics(results []queryResult, groundTruth [][]int32, topK int, wallSecs float64, docIDPrefix string) Metrics {
	if len(results) == 0 {
		return Metrics{}
	}

	latencies := make([]float64, len(results))
	var totalLat, recallSum float64
	prefixFmt := docIDPrefix + "-%d" // e.g. "vec-%d"

	for i, r := range results {
		ms := float64(r.latency.Microseconds()) / 1000.0
		latencies[i] = ms
		totalLat += ms

		// Recall: parse "<docIDPrefix>-<idx>" doc IDs and compare to ground truth
		if r.queryIdx < len(groundTruth) {
			gt := groundTruth[r.queryIdx]
			gtSet := make(map[int32]bool, topK)
			for j := 0; j < topK && j < len(gt); j++ {
				gtSet[gt[j]] = true
			}
			var hits int
			for _, id := range r.ids {
				var idx int32
				fmt.Sscanf(id, prefixFmt, &idx)
				if gtSet[idx] {
					hits++
				}
			}
			denom := topK
			if len(gtSet) < denom {
				denom = len(gtSet)
			}
			if denom > 0 {
				recallSum += float64(hits) / float64(denom)
			}
		}
	}

	sort.Float64s(latencies)
	n := len(results)
	p95idx := int(math.Min(float64(n-1), math.Ceil(0.95*float64(n))-1))
	p99idx := int(math.Min(float64(n-1), math.Ceil(0.99*float64(n))-1))

	return Metrics{
		TotalQueries:      n,
		SuccessfulQueries: n,
		ThroughputQPS:     float64(n) / wallSecs,
		LatencyAvgMs:      totalLat / float64(n),
		LatencyP95Ms:      latencies[p95idx],
		LatencyP99Ms:      latencies[p99idx],
		RecallAt10:        recallSum / float64(n),
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// §11  Main
// ─────────────────────────────────────────────────────────────────────────────

func main() {
	// Step 1: Scan os.Args for --config before flag.Parse()
	//         so we can load the INI before defining flag defaults.
	configPath := "config.ini"
	for i := 1; i < len(os.Args)-1; i++ {
		if os.Args[i] == "--config" || os.Args[i] == "-config" {
			configPath = os.Args[i+1]
			break
		}
	}

	// Step 2: Load INI
	c, err := loadConfig(configPath)
	if err != nil {
		log.Fatalf("Config error: %v", err)
	}

	// Step 3: Define all flags with INI values as defaults, then parse
	dropIndexOnly := parseFlags(c)

	log.Printf("index=%s  nprobes=%d  reranking=%v  topNScan=%d  workers=%d",
		c.Description, c.NProbes, c.Reranking, c.TopNScan, c.Workers)
	log.Printf("index_type=%s  vector_field=%s  scalar_fields=%v  include_fields=%v",
		c.IndexType, c.VectorField, c.ScalarFields, c.IncludeFields)
	opener := newDatasetOpener(c)

	scalarRows, err := loadJSONObjects(c.ScalarFieldsPath, opener, "scalar_fields")
	if err != nil {
		log.Fatalf("Failed to load scalar fields: %v", err)
	}
	queryFilters, err := loadJSONObjects(c.QueryFiltersPath, opener, "query_filters")
	if err != nil {
		log.Fatalf("Failed to load query filters: %v", err)
	}
	if len(scalarRows) > 0 {
		log.Printf("Loaded %d scalar field rows from %s", len(scalarRows), c.ScalarFieldsPath)
	}
	if len(queryFilters) > 0 {
		log.Printf("Loaded %d query filter rows from %s", len(queryFilters), c.QueryFiltersPath)
	}

	// Step 4: Connect to Couchbase
	cluster, err := connectCB(c)
	if err != nil {
		log.Fatalf("Connection failed: %v", err)
	}
	if dropIndexOnly {
		if err := dropIndex(cluster, c); err != nil {
			log.Fatalf("Drop index failed: %v", err)
		}
		log.Printf("Drop index complete.")
		return
	}

	// Step 5: Load data (parallel goroutines — no Python GIL limitation here)
	if !c.SkipLoad {
		log.Printf("Loading base vectors from %s ...", c.DataPath)
		if err := loadData(cluster, c, opener, scalarRows); err != nil {
			log.Fatalf("Failed to load base vectors: %v", err)
		}
	}

	// Step 6: Build index
	var buildTime float64
	if !c.SkipIndex {
		buildTime, err = createIndex(cluster, c)
		if err != nil {
			log.Fatalf("Index build failed: %v", err)
		}
	}

	// Step 7: Load query vectors and ground truth into RAM
	queryVecs, err := loadFvecs(c.QueryVectorsPath, opener)
	if err != nil {
		log.Fatalf("Failed to load query vectors: %v", err)
	}
	groundTruth, err := loadIvecs(c.GroundTruthPath, opener)
	if err != nil {
		log.Fatalf("Failed to load ground truth: %v", err)
	}
	// Trim to equal length
	n := len(queryVecs)
	if len(groundTruth) < n {
		n = len(groundTruth)
	}
	if len(queryFilters) > 0 {
		if len(queryFilters) < n {
			log.Fatalf("query_filters_path has %d rows but requires at least %d rows (one per query vector)", len(queryFilters), n)
		}
	}
	queryVecs = queryVecs[:n]
	groundTruth = groundTruth[:n]
	if len(queryFilters) > 0 {
		queryFilters = queryFilters[:n]
	}

	queryStmts := []string{buildQueryStmt(c, c.WhereClauseSQL)}
	if len(queryFilters) > 0 {
		queryStmts = make([]string, n)
		for i := 0; i < n; i++ {
			filterSQL, err := buildWhereExpr(queryFilters[i])
			if err != nil {
				log.Fatalf("Invalid query filter at index %d: %v", i, err)
			}
			whereSQL := combineWhereClauses(c.WhereClauseSQL, filterSQL)
			queryStmts[i] = buildQueryStmt(c, whereSQL)
		}
	}
	if len(c.ScalarFields) > 0 && strings.TrimSpace(c.WhereClauseSQL) == "" {
		log.Printf("Warning: scalar partition fields are configured (%v) but query.where_clause is empty; query may scan a much larger search space.", c.ScalarFields)
	}
	log.Printf("Query statement (sample): %s", queryStmts[0])

	// Step 8: Warmup phase (parallel goroutines — metrics discarded)
	if c.Warmup > 0 {
		log.Printf("Warming up for %ds (%d workers) ...", c.Warmup, c.Workers)
		runPhase(cluster, c, queryStmts, queryVecs, c.Warmup)
		log.Printf("Warmup complete.")
	}

	// Step 9: Measurement phase (parallel goroutines — closed-loop)
	log.Printf("Measuring for %ds (%d workers) ...", c.Duration, c.Workers)
	wallStart := time.Now()
	results := runPhase(cluster, c, queryStmts, queryVecs, c.Duration)
	wallSecs := time.Since(wallStart).Seconds()
	log.Printf("Measurement complete: %d queries in %.1fs", len(results), wallSecs)

	// Step 10: Compute metrics
	metrics := computeMetrics(results, groundTruth, c.TopK, wallSecs, c.DocIDPrefix)
	metrics.IndexBuildTimeS = buildTime

	log.Printf("recall=%.4f  qps=%.1f  avg=%.1fms  p95=%.1fms  p99=%.1fms",
		metrics.RecallAt10, metrics.ThroughputQPS,
		metrics.LatencyAvgMs, metrics.LatencyP95Ms, metrics.LatencyP99Ms,
	)

	// Step 11: Write JSON to stdout (Python subprocess reads this)
	//          Go's log package writes to stderr, so stdout is clean JSON only.
	out, _ := json.Marshal(metrics)
	fmt.Println(string(out))

	// Step 12: Also persist to file
	if c.OutputJSON != "" {
		if err := os.WriteFile(c.OutputJSON, out, 0644); err != nil {
			log.Printf("Warning: failed to write output file: %v", err)
		}
	}
}
