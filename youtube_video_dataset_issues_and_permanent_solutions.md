# YouTube Video Dataset --- Issues, Permanent Solutions & 41 GB Processed-Shard Remediation Plan

**Dataset:**
https://huggingface.co/datasets/raj270898/youtube-video-dataset\
**Purpose:** Remediate the audited dataset and make the
already-completed, processed \~41 GB shard collection reliable,
reproducible, and suitable for downstream video-understanding and
video-generation training.

> **Scope note:** This document is based on the earlier repository and
> release-report audit in this conversation. The reported 41.4 GB
> repository size and shard-level findings should be revalidated against
> the exact current repository revision before executing destructive
> operations. The solutions below are implementation recommendations,
> not claims that the repository has already been modified.

------------------------------------------------------------------------

## 1. Executive summary

The dataset has a useful annotation design: frames, clips, scenes,
sequences, events, temporal QA, long-video QA, and video descriptions.
The main risks are not simply dataset size. They are media availability,
rights provenance, stale root metadata, schema drift, incomplete quality
measurements, high review rates, temporal/camera-motion ambiguity, and
the absence of a reproducible training-ready export.

### Recommended order

1.  **Freeze and preserve the completed 41 GB processed data.** Do not
    rerun expensive processing or overwrite existing shards before
    checksums and manifests are recorded.
2.  **Audit media references and actual media availability.** Metadata
    rows are not equivalent to usable video/frame assets.
3.  **Generate one canonical manifest and statistics report from the
    actual shard files.**
4.  **Normalize schema versions and validate every row.**
5.  **Separate accepted, review, and rejected records; never silently
    promote review records.**
6.  **Validate temporal grounding, camera movement, subject movement,
    and source-video split isolation.**
7.  **Publish separate exports for video understanding, temporal
    reasoning, cinematic understanding, and video generation.**
8.  **Record provenance, rights status, checksums, and processing
    versions for every release.**

------------------------------------------------------------------------

## 2. Issue register with permanent solutions

Priority definitions:

-   **P0 --- Critical:** blocks trustworthy release or training.
-   **P1 --- High:** materially affects annotation quality or model
    performance.
-   **P2 --- Medium:** missing metadata or enrichment that should be
    added where verifiable.
-   **P3 --- Operational:** reproducibility, dataset management, and
    training-pipeline improvements.

### P0 --- Critical

  ----------------------------------------------------------------------------------------------
  ID                Issue             Permanent solution  Acceptance criteria
  ----------------- ----------------- ------------------- --------------------------------------

  P0-02             Referenced frame  Validate every      All frame records in a media-ready
                    images may be     frame path and      export resolve to decodable images.
                    missing           decode the image.   
                                      Check dimensions,   
                                      format, file size,  
                                      and checksum.       
                                      Re-extract only     
                                      missing/corrupt     
                                      frames from the     
                                      original source     
                                      when rights and     
                                      source availability 
                                      permit.             

  P0-03             Broken or         Use stable asset    Every clip/frame reference resolves
                    ambiguous         IDs rather than     through the manifest; no path depends
                    `clip_path` /     machine-specific    on a developer's local machine.
                    `frame_ids`       absolute paths.     
                    references        Maintain an         
                                      `assets` manifest   
                                      mapping IDs to      
                                      relative paths,     
                                      checksums, source   
                                      video IDs,          
                                      timestamps, and     
                                      media status.       

  P0-04             Root statistics   Rebuild             Recomputed totals match the manifest
                    disagree with     `statistics.json`   and exported files exactly.
                    shard totals      from the exact      
                                      release file list   
                                      using a single      
                                      deterministic       
                                      aggregation script. 
                                      Never hand-edit     
                                      totals. Include the 
                                      release ID and      
                                      source revision.    

  P0-05             Root manifest is  Generate            Manifest file list and checksums match
                    stale             `manifest.json`     the actual release directory.
                                      automatically after 
                                      every release.      
                                      Include every       
                                      shard, file size,   
                                      row count, schema   
                                      version, checksum,  
                                      split, and record   
                                      type.               

  P0-06             Multiple schema   Define a canonical  Every row in a published release
                    versions may be   versioned schema.   declares one supported schema version;
                    mixed             Add explicit        migrations are repeatable and tested.
                                      migration scripts   
                                      from every          
                                      supported older     
                                      version to the      
                                      canonical version.  
                                      Validate after      
                                      migration and       
                                      retain the original 
                                      source files.       

  P0-07             Original video    Maintain a          No media is included in a
                    rights are        rights/provenance   redistributed media package unless its
                    unclear           record per source   rights status permits that
                                      video: original     distribution.
                                      URL/ID,             
                                      creator/channel,    
                                      source license      
                                      where verified,     
                                      verification date,  
                                      allowed use,        
                                      redistribution      
                                      status, and         
                                      evidence URL.       

  P0-08             MIT metadata      Clearly separate    Users can identify the license and
                    license may be    licenses for code,  permitted use of each asset class
                    confused with     annotations,        without assuming the metadata license
                    source-media      metadata,           covers source media.
                    rights            transcripts,        
                                      frames, and video   
                                      assets. Add a       
                                      rights notice to    
                                      the dataset card    
                                      and each release    
                                      manifest.           
  ----------------------------------------------------------------------------------------------

### P1 --- High priority

  ------------------------------------------------------------------------------------------
  ID                Issue                     Permanent solution      Acceptance criteria
  ----------------- ------------------------- ----------------------- ----------------------
  P1-01             Frame/clip/scene          Add nullable grounding  Every score includes
                    grounding scores missing  fields with a           method/version and is
                                              documented scoring      not fabricated from
                                              method. Calculate only  unrelated quality
                                              when evidence exists;   scores.
                                              use `null` when not     
                                              evaluated.              

  P1-02             Temporal accuracy         Validate start/end      All media-backed
                    measurements missing      timestamps against      temporal records pass
                                              source duration and     range and ordering
                                              decoded media. Add      checks or are flagged.
                                              boundary error,         
                                              timestamp confidence,   
                                              and validation status.  

  P1-03             Large review population   Create reason-coded     Review rate decreases
                                              review queues.          through measured
                                              Prioritize by failure   fixes; every status
                                              reason and impact;      change has an audit
                                              repair upstream         trail.
                                              extraction/annotation   
                                              prompts rather than     
                                              merely changing status. 

  P1-04             Most sequences require    Validate sequence       Sequence order and
                    review                    ordering, clip overlap, temporal relationships
                                              continuity, and         are valid; unresolved
                                              source-video            cases remain in
                                              membership. Rebuild     review.
                                              sequences from          
                                              validated scene/clip    
                                              boundaries where        
                                              needed.                 

  P1-05             Video descriptions have   Rework description      Descriptions are
                    very high review rate     generation with         grounded in visible
                                              structured visual       evidence and pass a
                                              evidence, timestamped   documented review
                                              observations, and       sample.
                                              hallucination checks.   
                                              Compare descriptions to 
                                              sampled frames and      
                                              clips.                  

  P1-06             Long-video QA lacks       Require answer evidence Each accepted
                    temporal accuracy         intervals and           long-video QA item has
                                              referenced events.      valid evidence and a
                                              Check that evidence     verified temporal
                                              intervals fall within   range.
                                              the video and support   
                                              the answer.             

  P1-07             Human verification not    Introduce stratified    Release report states
                    run                       human review by record  sample size, reviewer
                                              type, source video,     protocol, agreement,
                                              quality tier, and       and unresolved
                                              failure category. Use   disagreement rate.
                                              double annotation for a 
                                              calibration subset and  
                                              adjudication for        
                                              disagreements.          

  P1-08             Heuristic verification is Keep heuristic checks   Structural and
                    not independent semantic  for structural          semantic validation
                    verification              validation; add a       are reported
                                              separate semantic       separately.
                                              review stage using      
                                              human reviewers or      
                                              independently validated 
                                              evaluators.             

  P1-09             Pan/tracking/dolly/zoom   Define a camera-motion  Confusion matrix is
                    ambiguity                 taxonomy with           measured on a
                                              operational definitions human-labelled test
                                              and positive/negative   set; uncertain labels
                                              examples. Use motion    remain unknown/review.
                                              estimation and          
                                              camera-pose evidence;   
                                              send ambiguous clips to 
                                              review.                 

  P1-10             Subject movement confused Annotate camera motion  Records distinguish
                    with camera movement      and subject motion as   world/camera/subject
                                              separate entities. Use  motion and do not
                                              object tracks and       infer camera movement
                                              camera-motion           from subject
                                              estimates; represent    displacement alone.
                                              relative motion         
                                              explicitly.             

  P1-11             Event boundaries and      Add automated timestamp No accepted event has
                    temporal relations need   range/order checks and  invalid timestamps;
                    verification              a sampled human         boundary quality is
                                              boundary audit. Record  measured on a reviewed
                                              boundary source and     sample.
                                              confidence.             

  P1-12             Cinematic subset may be   Report actual eligible  Dataset card reports
                    too small for broad       unique clips and hours  counts and duration by
                    generation claims         by movement, shot size, cinematic category; no
                                              lighting, and scene     unsupported coverage
                                              type. Expand only with  claims.
                                              rights-cleared,         
                                              quality-checked media.  

  P1-13             Clip boundary failures    Re-extract failed clips Failed clips are
                                              from source timestamps; repaired or excluded;
                                              verify decoded          release report lists
                                              duration, frame count,  residual failures.
                                              start/end tolerance,    
                                              and audio/video stream  
                                              integrity.              

  P1-14             Long-video QA acceptance  Analyze                 Accepted QA set has
                    is low                    rejection/review        evidence-backed
                                              reasons; improve        answers and a measured
                                              question generation and error rate on a
                                              evidence selection.     held-out
                                              Remove ambiguous,       human-reviewed sample.
                                              unanswerable, or weakly 
                                              grounded questions.     

  P1-15             Video-description         Link each description   Accepted descriptions
                    grounding is incomplete   to frame/clip evidence  have resolvable
                                              IDs and timestamps.     evidence references or
                                              Separate observed facts explicitly declare
                                              from inferred           that grounding was not
                                              interpretation.         assessed.
  ------------------------------------------------------------------------------------------

### P2 --- Annotation and metadata improvements

  ------------------------------------------------------------------------------------
  ID                Issue             Permanent solution           Acceptance criteria
  ----------------- ----------------- ---------------------------- -------------------
  P2-01             Camera pose is    Add camera-pose fields only  Pose metadata
                    not consistently  where estimated with a       includes
                    available         documented method and        estimator/version
                                      confidence. Keep unknown     and confidence.
                                      values null.                 

  P2-02             Focal length is   Populate focal length only   Unknown focal
                    unavailable or    from trustworthy source      length remains
                    inferred          metadata or verified         null; provenance is
                                      production metadata. Do not  recorded for known
                                      guess from appearance.       values.

  P2-03             Stabilization     Add `tripod`, `handheld`,    Every non-unknown
                    type missing      `gimbal`, `vehicle_mount`,   label has evidence
                                      `drone`, `unknown` labels    and confidence.
                                      only when visually or        
                                      externally supported.        

  P2-04             Focus transitions Annotate focus pulls/rack    Focus transition
                    missing           focus only when visible and  has start/end
                                      temporally localizable.      timestamps and
                                                                   evidence.

  P2-05             Persistent object Use multi-object tracking to Track IDs are
                    IDs missing       assign IDs across frames,    stable within a
                                      then validate identity       video; uncertain
                                      switches on a sample.        associations are
                                                                   flagged.

  P2-06             Lighting metadata Define controlled labels for Labels use a
                    incomplete        source, direction, softness, documented taxonomy
                                      and time-of-day appearance.  and confidence.
                                      Distinguish visible          
                                      appearance from inferred     
                                      physical setup.              

  P2-07             Color metadata    Record color space, transfer Measurements
                    incomplete        function, range, white       specify color space
                                      balance only when known,     and calculation
                                      plus measured image          method.
                                      statistics where             
                                      appropriate.                 

  P2-08             Audio alignment   Preserve word-level          Accepted transcript
                    incomplete        timestamps, speaker IDs,     segments align to
                                      language, and alignment      the source audio
                                      confidence where available.  within a defined
                                      Validate timestamps against  tolerance.
                                      audio duration.              

  P2-09             OCR grounding     Store recognized text with   OCR entries resolve
                    incomplete        bounding boxes, frame        to the correct
                                      timestamps, OCR              frame and bounding
                                      engine/version, and          box.
                                      confidence.                  

  P2-10             Motion            Store                        Measurements are
                    measurements need optical-flow/camera-motion   reproducible and
                    validation        method, coordinate           not mixed across
                                      convention, units, and       incompatible
                                      confidence. Validate on      coordinate systems.
                                      controlled clips.            

  P2-11             Annotation        Add annotator/model name,    Each generated
                    provenance        model version,               annotation can be
                    incomplete        prompt/template version,     traced to its
                                      processing timestamp, and    pipeline and model
                                      pipeline commit.             version.

  P2-12             Confidence scores Calibrate confidence against Confidence values
                    may be            a human-labelled validation  are empirically
                    uncalibrated      set. Report reliability and  calibrated or
                                      calibration metrics by task. explicitly labelled
                                                                   uncalibrated.
  ------------------------------------------------------------------------------------

### P3 --- Dataset operations and training

  --------------------------------------------------------------------------------
  ID                Issue              Permanent solution Acceptance criteria
  ----------------- ------------------ ------------------ ------------------------
  P3-01             Duplicate records  Choose one         Training exports contain
                    may enter from     canonical source   no unintended duplicate
                    root exports and   of truth.          IDs or duplicate media
                    shards             Deduplicate by     samples.
                                       stable record ID   
                                       and also detect    
                                       content duplicates 
                                       using normalized   
                                       hashes.            

  P3-02             Root metadata not  Make release       A release cannot publish
                    synchronized       generation a       if generated metadata
                                       single automated   disagrees.
                                       pipeline that      
                                       writes shards,     
                                       manifest,          
                                       statistics, card   
                                       summary, and       
                                       checksums          
                                       together.          

  P3-03             Train/test leakage Split by source    No source video appears
                    risk               video (and         across
                                       preferably         train/validation/test;
                                       channel/creator    report leakage checks.
                                       where              
                                       appropriate), not  
                                       by individual      
                                       frame or clip.     
                                       Keep all derived   
                                       assets from a      
                                       source in one      
                                       split.             

  P3-04             Quality filters    Define explicit    Every training export
                    inconsistent       status policy:     records its filters and
                                       `accepted`,        counts by status/reason.
                                       `review`,          
                                       `rejected`;        
                                       document           
                                       eligibility by     
                                       task.              

  P3-05             No robust          Create held-out    Benchmark has fixed
                    benchmark          human-verified     examples, annotation
                                       evaluation sets    guidelines, and
                                       for captioning,    versioned metrics.
                                       temporal QA,       
                                       camera motion, and 
                                       cinematic          
                                       understanding.     

  P3-06             Captions may not   Create             Generation export has
                    be generation      task-specific      verified prompt-target
                    prompts            prompt/caption     pairs and clear task
                                       pairs grounded in  definitions.
                                       the actual clip.   
                                       Keep descriptive   
                                       captions distinct  
                                       from instruction   
                                       prompts.           

  P3-07             Continuity         Add shot/scene     Continuity links resolve
                    annotations        relationship       and are validated within
                    incomplete         fields: same       each source video.
                                       scene, temporal    
                                       adjacency, subject 
                                       continuity,        
                                       location           
                                       continuity, and    
                                       transition type    
                                       where supported.   

  P3-08             Cinematic          Produce coverage   Release reports expose
                    diversity unknown  dashboards by      distribution and
                                       camera movement,   underrepresented
                                       shot scale, angle, categories.
                                       lighting, scene,   
                                       duration, and      
                                       source.            

  P3-09             Checksums not      Calculate SHA-256  All released files have
                    enforced           for every asset    checksums and integrity
                                       and dataset file;  checks pass.
                                       verify after       
                                       upload/download.   

  P3-10             No dedicated       Publish separate   Users can load each task
                    generation-ready   task-specific      export without mixing
                    export             datasets and a     incompatible record
                                       media-ready export types.
                                       that includes      
                                       accessible assets  
                                       or resolvable      
                                       licensed asset     
                                       references.        
  --------------------------------------------------------------------------------

------------------------------------------------------------------------

## 3. Permanent architecture for the completed \~41 GB processed shards

### 3.1 Do not reprocess the 41 GB by default

If the 41 GB represents already-completed processing, preserve it as an
immutable input snapshot. Reprocessing everything can waste compute,
introduce new nondeterministic annotations, and make it difficult to
compare old and new results.

Use this process instead:

1.  Freeze the current files.
2.  Inventory every file and shard.
3.  Compute checksums.
4.  Build a manifest and row-count report.
5.  Validate schema and references.
6.  Run targeted repair only on failed records.
7.  Write repaired output to a **new release directory**.
8.  Compare old and new releases.
9.  Publish only after all release gates pass.

Never overwrite the only copy of the processed shards.

### 3.2 Recommended storage layout

``` text
youtube-video-dataset/
│
├── raw/                         # Original inputs; immutable
│
├── processed/                   # Existing completed ~41 GB snapshot
│   ├── shard_0001/
│   ├── shard_0002/
│   ├── ...
│   └── shard_0014/
│
├── manifests/
│   ├── processed_manifest.json
│   ├── assets_manifest.parquet
│   ├── record_counts.json
│   ├── checksums.sha256
│   └── validation_summary.json
│
├── schemas/
│   ├── canonical.schema.json
│   └── migrations/
│
├── staging/
│   ├── repair_candidates/
│   └── repaired_records/
│
├── releases/
│   ├── v2.2.1/
│   │   ├── understanding/
│   │   ├── temporal_reasoning/
│   │   ├── cinematic/
│   │   ├── generation_ready/
│   │   ├── manifest.json
│   │   ├── statistics.json
│   │   └── checksums.sha256
│   └── ...
│
└── reports/
    ├── media_integrity.csv
    ├── schema_errors.csv
    ├── duplicate_records.csv
    ├── temporal_errors.csv
    ├── rights_review.csv
    └── release_audit.md
```

The existing shard folders can remain as they are. The structure above
is a target organization; do not move or rename existing data until the
inventory and references have been verified.

### 3.3 Inventory and checksum the existing files

Create a file inventory with:

-   Relative path
-   File size in bytes
-   SHA-256
-   File extension
-   Shard ID
-   Detected format
-   Row count (for Parquet/JSONL)
-   Schema version
-   Record type

Example checksum command:

``` bash
find processed -type f -print0 \
  | sort -z \
  | xargs -0 sha256sum > manifests/checksums.sha256
```

For a portable manifest, use relative paths. Do not embed a
machine-specific path such as `/Users/name/...`.

### 3.4 Validate the shards without loading all 41 GB into RAM

Use streaming/chunked processing:

-   JSONL: read one line at a time.
-   Parquet: inspect metadata first, then scan in batches.
-   Images/video: validate in bounded worker pools.
-   Reports: write incrementally to CSV/Parquet.
-   Avoid concatenating every shard into one in-memory DataFrame.

Suggested tools:

  Task                               Suggested tool
  ---------------------------------- --------------------------------
  Parquet metadata and scanning      PyArrow
  Dataframe checks                   Polars
  JSONL streaming                    Python standard library
  Schema validation                  JSON Schema / Pydantic
  Video metadata and decode checks   FFprobe / FFmpeg
  Image validation                   Pillow / OpenCV
  Checksums                          SHA-256
  Duplicate detection                Stable IDs + normalized hashes
  Reports                            CSV / Parquet / Markdown

### 3.5 Use a canonical record schema

Normalize all supported versions into one canonical schema. Preserve
original fields in an `extensions` object if they cannot yet be mapped.

Recommended common fields:

``` json
{
  "record_id": "stable_unique_id",
  "record_type": "clip",
  "schema_version": "2.2.1",
  "source_video_id": "source_id",
  "asset_id": "asset_id_or_null",
  "split": "train",
  "quality_status": "accepted",
  "quality_reasons": [],
  "start_time": 12.4,
  "end_time": 18.7,
  "evidence_ids": [],
  "annotation_provenance": {
    "pipeline_version": "version",
    "model_version": "version_or_null",
    "prompt_version": "version_or_null"
  },
  "rights": {
    "status": "unverified",
    "license": null,
    "verified_at": null
  }
}
```

This is a suggested common envelope, not a replacement for the
task-specific fields. Keep frame, clip, scene, event, and QA payloads
type-specific.

### 3.6 Add a media asset registry

Do not rely only on paths embedded in records. Create one asset
registry:

``` json
{
  "asset_id": "asset_000001",
  "source_video_id": "source_001",
  "asset_type": "video_clip",
  "relative_path": "media/clips/asset_000001.mp4",
  "sha256": "computed_sha256",
  "size_bytes": 12345678,
  "duration_seconds": 6.3,
  "width": 1920,
  "height": 1080,
  "video_codec": "h264",
  "media_status": "available",
  "rights_status": "unverified"
}
```

Only populate measured metadata after inspecting the actual file. If the
original asset is unavailable, set `media_status` to `missing` and keep
the annotation record separate from media-dependent training exports.

------------------------------------------------------------------------

## 4. Quality-status policy for the existing processed data

Do not delete every record marked `review`. Keep it in the archive and
separate it from the default training exports.

  -----------------------------------------------------------------------
  Status                  Meaning                 Default training policy
  ----------------------- ----------------------- -----------------------
  `accepted`              Passed the defined      Eligible, subject to
                          checks for this task    task-specific filters

  `review`                Uncertain, incomplete,  Exclude from default
                          or failed a non-fatal   training; retain for
                          check                   repair/review

  `rejected`              Invalid, contradictory, Exclude from training
                          unusable, or            
                          rights-ineligible       

  `not_evaluated`         A check was not run     Do not interpret as
                                                  passed
  -----------------------------------------------------------------------

A status must be task-specific where appropriate. A scene annotation
might be acceptable for scene classification but not acceptable for
video generation if the actual media is missing.

Never convert missing scores to `0`, `1`, or an average. Use `null` plus
an explicit status.

------------------------------------------------------------------------

## 5. Repair strategy that avoids rerunning all processing

Use targeted repair rather than a full pipeline rerun.

### Step A --- Create repair queues

Generate separate queues:

-   Missing/corrupt media
-   Missing/corrupt frames
-   Invalid timestamps
-   Broken evidence references
-   Schema migration failures
-   Duplicate records
-   Low-confidence camera movement
-   Unverified descriptions
-   Rights-unverified source media

Each queue should include `record_id`, `source_video_id`, shard, failure
reason, and proposed repair action.

### Step B --- Repair only the affected records

Examples:

-   If a frame is missing but the source video is available and
    permitted, re-extract that frame.
-   If a timestamp is invalid, re-evaluate the relevant source segment.
-   If a schema field was renamed, apply a deterministic migration.
-   If camera movement is ambiguous, mark it `unknown` or send it for
    review; do not guess.
-   If rights are unclear, keep the annotation metadata but block
    redistribution of the media.

### Step C --- Preserve lineage

Every repaired record should include:

-   Original record ID
-   New record ID or revision
-   Repair reason
-   Repair method
-   Pipeline version
-   Timestamp of repair
-   Previous checksum / new checksum where applicable
-   Reviewer or process identifier

### Step D --- Compare releases

Compare old and new versions by:

-   Record counts by type
-   Unique source videos
-   Unique record IDs
-   Accepted/review/rejected counts
-   Missing media count
-   Temporal validation failures
-   Duplicate count
-   Schema validation failures
-   Split leakage
-   File checksums

Do not publish if unexplained record loss or count changes occur.

------------------------------------------------------------------------

## 6. Separate training exports

Do not combine all record types into one undifferentiated training file.

### A. Video understanding export

Include: - Media-backed frames/clips - Scenes - Events - Grounded
descriptions - Evidence references - Verified visual attributes

### B. Temporal reasoning export

Include: - Temporal QA - Long-video QA - Event ordering -
Before/after/overlap relations - Evidence intervals - Source-video-level
split

### C. Cinematic understanding export

Include: - Verified camera movement - Shot size and angle - Subject
movement - Camera-relative motion - Lighting and color attributes -
Actual clip/frame references

### D. Video-generation-ready export

Require: - Rights-cleared media or a clearly permitted media access
route - Actual decodable video clips - Matching frame sequences -
Verified captions or prompts - Clip duration and frame rate - Resolution
and aspect ratio - Camera and subject motion labels where supported -
Scene/shot continuity where supported - Source-video-level split
isolation

**Important:** a metadata-only record can support some text-only
experiments, but it is not a substitute for the actual video
input/target required by a video-generation training pipeline.

------------------------------------------------------------------------

## 7. Release automation and quality gates

Make the release pipeline fail automatically if critical checks fail.

### Required release gates

  Gate              Required result
  ----------------- ------------------------------------------------
  File inventory    Every release file is listed
  Checksums         All checksums verify
  Schema            No unhandled schema errors
  Record IDs        No duplicate IDs
  References        All required references resolve
  Media             Every media-dependent export asset is readable
  Timestamps        Valid ranges and ordering
  Evidence          QA evidence references resolve
  Splits            No source-video leakage
  Rights            Media redistribution policy is satisfied
  Statistics        Manifest and statistics agree
  Reproducibility   Pipeline and schema versions are recorded

A release should have a clear status such as `passed`,
`passed_with_exclusions`, or `blocked`. A report must disclose all
exclusions and unresolved issues.

------------------------------------------------------------------------

## 8. Suggested implementation phases

  -----------------------------------------------------------------------
  Phase                   Work                    Deliverable
  ----------------------- ----------------------- -----------------------
  1\. Preserve            Freeze current 41 GB    Immutable baseline
                          snapshot; inventory and 
                          checksum                

  2\. Inspect             Scan Parquet/JSONL      Shard inventory
                          metadata, row counts,   
                          schema versions         

  3\. Integrity           Check IDs, references,  Validation reports
                          media, timestamps, and  
                          splits                  

  4\. Normalize           Migrate to canonical    Normalized staging data
                          schema                  

  5\. Repair              Repair only flagged     Repaired release
                          records                 candidate

  6\. Verify              Human sample review and Quality report
                          automated regression    
                          checks                  

  7\. Export              Build task-specific     Four training exports
                          datasets                

  8\. Publish             Generate card,          Versioned release
                          manifest, statistics,   
                          checksums, rights notes 
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## 9. Definition of done

The 41 GB processed dataset can be considered operationally ready for a
specific training task only when:

-   The exact processed snapshot is preserved and checksummed.
-   All shards are inventoried and their formats/row counts are known.
-   A single canonical schema is applied or migrations are documented.
-   Root statistics and manifests are generated from the actual release
    files.
-   Every media-dependent record has a valid asset or is excluded from
    that export.
-   Invalid timestamps and broken evidence references are repaired or
    excluded.
-   Review records are not silently treated as accepted.
-   Train/validation/test splits are isolated by source video.
-   Rights status is documented and compatible with the intended use.
-   The relevant task-specific export passes its own quality gates.
-   The release report states limitations and unresolved issues.

### Final recommendation

**Preserve the completed 41 GB; do not rerun the entire processing
pipeline.** First build an inventory, checksums, canonical manifest, and
validation report. Then repair only failed records and produce a new
versioned release. Keep the original processed shards unchanged so the
new release can be compared and rolled back if necessary.

This approach protects the work already completed, avoids unnecessary
compute, and creates a traceable path from the existing processed
dataset to reliable training exports.
