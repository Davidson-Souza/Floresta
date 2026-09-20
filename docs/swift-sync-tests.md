# SwiftSync performance experiments

This document records observations from the non-assumevalid SwiftSync proof of concept and provides a repeatable workflow for evaluating future changes. Measurements below were taken on an 8-logical-CPU AMD Ryzen 5 2500U system with about 9.5 GiB of RAM and ext4 storage.

The implementation was changing while these measurements were taken. Treat every number as a point-in-time result, not a stable benchmark. Always record the binary build ID, source revision, height range, and configuration with a new result.

## Current design

SwiftSync has two concurrent phases:

1. **Indexing** performs context-free block checks and inserts every hinted-spent output into a temporary `floresta-db` map.
2. **Validation** pops those prevouts from the map, reconstructs scripts, and performs contextual consensus and script validation.

Blocks may be indexed in any order. A block becomes validation-eligible only after every preceding height has been indexed. Validation itself may run out of order once this frontier requirement is satisfied.

The current proof of concept uses `batch_pop` in chunks capped by a 16,384-entry reclamation watermark. `floresta-db` does not provide hazard pointers or reader epochs, so reclamation can still race another worker retaining an offset in the same bucket. Any detected database corruption aborts SwiftSync. This is acceptable only for the current experiment.

Relevant current settings:

- Validation/indexing worker capacity: 16.
- Host logical CPUs used in these tests: 8.
- Maximum network-inflight blocks: 100.
- Download window: byte based, initially and maximally 1 GiB, with a 64 MiB minimum.
- Unknown block reservation: EWMA initialized at 1 MiB and bounded to 1 KiB–4 MiB.
- Total requested plus buffered blocks: at most 2,000.
- Allocator page size: 1 MiB for both `body` and `blobs`.
- Reclamation watermark: 16,384 prevouts per `batch_pop`.
- Body key: 96 least-significant txid bits plus 32-bit `vout`, packed into 16 bytes.

## CPU observations

### Script validation is the steady-state CPU cost

A 15-second `perf stat` capture during a healthy, busy period measured:

- 111.17 CPU-seconds over 15.05 wall-seconds.
- Approximately 7.38 of 8 logical CPUs busy.
- 179.4 billion cycles.
- 265.4 billion instructions.
- IPC approximately 1.48.
- 52,863 context switches.

A later 20-second sample measured approximately 7.13 busy CPUs and 127,464 context switches.

A 40,082-sample CPU profile attributed approximately:

- 90.8% to the `florestad` executable.
- 5.2% to libc.
- The remainder primarily to unresolved kernel code.

The hottest instruction region resolved into `libsecp256k1` field/group arithmetic. This agrees with the application timings: contextual validation, particularly signature verification, is the dominant steady-state CPU consumer.

Around height 200,000, saturated five-second windows generally showed:

| Metric | Typical range |
|---|---:|
| Prevout fetch | 4–7 ms/block |
| Consensus validation | 227–341 ms/block |
| Busy workers | 32/32 |
| Queued blocks | 850–960 |
| Processed blocks | 400–670 per 5 seconds |
| Processed throughput | about 70–90 Mbps |

Across 20 saturated windows, median prevout fetch was about 5.8 ms and median consensus validation was about 293 ms. At that point the UTXO database was not the primary limiter.

### Worker oversubscription

The process had approximately 168 OS threads while SwiftSync exposed 32 worker slots on an 8-CPU machine. Tokio's blocking pool retains worker threads, so thread count is larger than the explicit SwiftSync capacity.

This creates scheduler and allocator overhead and increases tail latency, but an A/B test is required before reducing the worker count. Extra concurrency also hides mmap and random-read latency. Candidate CPU-worker counts to test are 8, 12, 16, and 32.

### Accidental transaction-size hot loop

One profile matched its symbolized binary exactly and reported:

```text
96.62%  Transaction::total_size
 2.93%  UtreexoNode::unprocessed_block_bytes
```

The byte-based download admission path recalculated the complete serialized size of every buffered block for every five-block GETDATA refill. `Block::total_size()` walks all transactions, inputs, outputs, scripts, and witnesses. With thousands of buffered blocks this made window filling approximately quadratic.

This was fixed by maintaining a cached byte counter:

- Increment once when an accepted block enters SwiftSync.
- Preserve the charge while moving from indexing to validation.
- Decrement after validation or failed indexing.
- Read the counter in constant time during request admission.

A new profile is required to quantify the improvement. `Transaction::total_size` should no longer be a visible admission-path hotspot.

## Pipeline starvation

Partial ordering can turn one delayed block into a global validation stall. One observed sequence was:

```text
buffered_blocks=1014 queued_blocks=0 busy_workers=0 oldest_request_age=13.8s
buffered_blocks=1015 queued_blocks=0 busy_workers=0 oldest_request_age=33.8s
queued_blocks=956 busy_workers=32 avg_worker_turnaround_ms=30131
```

Later blocks had been indexed, but one missing height prevented them from becoming eligible. When the missing block arrived, almost the entire buffered run became eligible at once.

A separate 83.75-second cycles profile showed the same burst shape:

- Active for roughly the first 25 seconds.
- Almost idle for two 5–10 second intervals.
- Approximately 72% of all cycles in the final ten seconds.

A cycles-only profile cannot identify the off-CPU wait directly, but this pattern is consistent with frontier starvation.

SwiftSync now derives block request timeouts from ready peers' average response-time EMA:

```text
timeout = clamp(4 × average_peer_time + 2 seconds, 1 second, 30 seconds)
```

With no samples, the fallback average is two seconds, producing a ten-second timeout. Timed-out SwiftSync block peers are excluded before selecting a replacement and are not penalized. Logs include `dynamic_timeout_ms` and `average_peer_ms`.

Further frontier-specific work may still be useful:

- Explicitly track the lowest missing indexed height.
- Reserve request slots for that height.
- Permit redundant requests for the frontier block.
- Avoid allowing far-ahead blocks to consume every request slot.

## Download-window observations

The byte window initially caused an unbounded GETDATA loop when the estimated early-block size became small. `MAX_INFLIGHT_REQUESTS` existed but was not enforced by SwiftSync's admission path. Peers disconnected and the node repeatedly logged:

```text
Failed to request blocks: ChannelSend(SendError { .. })
```

The byte window and network-inflight limit are now independent:

- Up to the current byte cap may be downloaded or buffered.
- At most 100 blocks may be outstanding on the network.
- Freed network slots are refilled as replies arrive.

An actual block can exceed its reservation and temporarily push the byte count over the cap. New requests stop until the backlog drains.

### OOM incident and safeguards

One run allowed the adaptive window to reach 2 GiB and accumulated 21,106 pending
blocks, including 21,063 decoded blocks and 20,830 queued blocks. The window charged
serialized bytes, not the substantially larger decoded Rust object graph. Systemd later
reported an 8.4 GiB memory peak, a 3.1 GiB swap peak, and a kernel OOM kill. Even after
the visible backlog drained, allocator arenas and database page pressure kept the process
thrashing; prevout-pop latency reached multiple seconds.

The safeguards added after this incident are:

- The serialized byte window can no longer grow beyond 1 GiB.
- Requested plus downloaded blocks are independently capped at 2,000.
- Upward window probes are disabled while every worker is occupied and work is queued.
- An active upward trial is rejected if worker pressure appears before it completes.
- Periodic diagnostics report `rss_mib`, `rss_anon_mib`, `rss_file_mib`, and `swap_mib`.

## UTXO map layout and churn

### Compact values inline when representable

SwiftSync first packs reconstructable compact leaves into one little-endian 64-bit value:

| Field | Bits |
|---|---:|
| Amount | 37 |
| `ScriptPubKeyKind` | 3 |
| Coinbase flag | 1 |
| Creation height | 23 |
| **Total** | **64** |

`floresta-db` automatically inlines an eight-byte value when its reserved high tag bit is clear. At current chain heights the packed height leaves that bit clear, so the common reconstructable UTXO stays entirely in the 32-byte body node.

SwiftSync falls back to the extended blob encoding when the script is not reconstructable, the height exceeds 23 bits, or the amount exceeds 37 bits. That encoding retains creation median time past and the truncated script commitment. The database also automatically sends any packed word with its tag bit set to the blob allocator.

The packed representation reconstructs `creation_time` as zero; SwiftSync's current validation path does not consume that field.

### Value access path

Every validation input is processed through:

```text
batch_pop compact OutPoint, in chunks of at most 16,384
  -> traverse body buckets in locality order
  -> return packed inline values directly
  -> sort and read fallback blob offsets
  -> unlink and release body/blob allocations
  -> restore values to request order
  -> unpack the compact leaf or decode its fallback representation
  -> reconstruct scripts
```

Indexing therefore avoids blob allocation for the common compact leaf while retaining the extended representation for values that cannot be packed safely.

A sampled address, `0x6046d473c970`, resolved to:

```text
mapping: /home/work/.floresta/.swiftsync-utxos-17118024126200411434/blobs
offset:  0x58d3c970 = 1,490,274,672 bytes
```

At the time, the logical blob file was approximately 1.76 GiB, so this was an active file offset rather than unused virtual address space.

One `smaps` snapshot showed:

| Mapping | Resident pages |
|---|---:|
| `body` | about 1.35 GiB |
| `blobs` | about 22.9 MiB |

The body index was largely resident while only a small portion of the blob value space was resident. Random historical prevout access can therefore produce frequent blob faults. Builds or other filesystem activity can evict this cache and distort measurements; do not compile large targets while collecting IBD data.

### Reclamation behavior

The allocator is bump-only within an open block. Deleting an object decrements the block's live count. A sealed block enters the free list only after its live count reaches zero. Individual holes in a partially live block are not reused.

At a 1 MiB allocator block size:

| File | Approximate objects per allocator block |
|---|---:|
| Body, 72-byte fixed nodes | 14,563 |
| Blobs, common 32-byte stride | 32,768 |

A single long-lived object can therefore pin many deleted allocations. Smaller blocks improve whole-block reclamation but increase allocator metadata, free-list traffic, growth calls, and potentially physical extent fragmentation.

A better long-term design separates physical growth from logical reclamation:

```text
32–64 MiB filesystem super-extents
  -> 64–256 KiB allocator slabs
    -> fixed 72-byte body slots
```

Hazard pointers or epochs could retire and reuse individual fixed-size body slots safely. Tagged offsets or generations are required to prevent ABA when an offset is reused.

## Filesystem fragmentation snapshot

The live ext4 snapshot was:

| File | Size | Physical extents | Extents/GiB | Average extent |
|---|---:|---:|---:|---:|
| `body` | 4.310 GiB | 1,965 | 456.0 | 2.246 MiB |
| `blobs` | 1.960 GiB | 447 | 228.1 | 4.490 MiB |

The body was roughly twice as fragmented per GiB. Counts changed slightly while the process was active, as expected.

The database grows by request-sized runs of 1 MiB allocator pages. Ext4 may merge neighboring allocations into larger physical extents. Reclaiming a page does not punch a filesystem hole or reduce existing extent counts; a SIMD count scan makes zero-count pages available for reuse.

Reducing allocator page size has opposing effects:

- **Better internal reuse:** fewer unrelated objects must disappear before a page is reusable.
- **Potentially worse physical fragmentation:** more growth operations and smaller allocation requests.
- **No change to fault granularity:** mmap still faults ordinary filesystem pages, normally 4 KiB.

A 256 KiB allocator page is the first candidate to test. It gives four-times-finer reclamation without the 16-times growth/metadata increase of the 64 KiB minimum.

## Repeatable evaluation commands

### Build an optimized binary with symbols

```bash
CARGO_PROFILE_RELEASE_DEBUG=2 \
CARGO_PROFILE_RELEASE_STRIP=none \
  cargo build --release -p florestad

file target/release/florestad
```

Record the build ID before profiling:

```bash
file target/release/florestad
perf buildid-list -i perf.data
```

The capture and executable build IDs must match for exact symbol attribution.

### Identify the running daemon

```bash
PID="$(pgrep -n florestad)"
printf 'pid=%s\n' "$PID"
```

### CPU counters

```bash
perf stat -p "$PID" \
  -e task-clock,context-switches,cpu-migrations,page-faults,major-faults,minor-faults,cycles,instructions,branches,branch-misses,cache-references,cache-misses \
  -- sleep 30
```

Useful derived values:

- Busy CPUs: `task-clock / wall time`.
- IPC: `instructions / cycles`.
- Cache miss ratio: `cache-misses / cache-references`.
- Context switches per second.

### On-CPU profile

```bash
perf record \
  -F 199 \
  --call-graph dwarf \
  -p "$PID" \
  -o "perf-$(date +%Y%m%d-%H%M%S).data" \
  -- sleep 30
```

```bash
perf report -i perf-YYYYMMDD-HHMMSS.data \
  --stdio --no-children --sort dso,symbol --percent-limit 0.25
```

Do not overwrite a useful baseline with a new `perf.data`; use timestamped filenames.

### Activity over time

This distinguishes continuous saturation from frontier starvation:

```bash
perf report -i perf-YYYYMMDD-HHMMSS.data \
  --stdio --no-children --sort time,dso \
  --time-quantum 5s --percent-limit 1
```

### Off-CPU scheduling

A cycles-only profile cannot explain why workers are asleep. If permissions allow:

```bash
sudo perf sched record -p "$PID" -- sleep 30
perf sched timehist
```

Use this when CPU samples show long idle intervals despite a large buffered backlog.

### Process I/O

Capture `/proc` counters before and after a fixed interval:

```bash
cat "/proc/$PID/io"
sleep 30
cat "/proc/$PID/io"
```

Important fields are `read_bytes`, `write_bytes`, `syscr`, and `syscw`. Compute differences per second rather than comparing cumulative values.

### Memory mappings

```bash
cat "/proc/$PID/maps"
cat "/proc/$PID/smaps"
```

To resolve an observed virtual address to a mapping and file offset:

```bash
python3 - "$PID" 0xADDRESS <<'PY'
from pathlib import Path
import sys

pid = int(sys.argv[1])
address = int(sys.argv[2], 0)
for line in Path(f"/proc/{pid}/maps").read_text().splitlines():
    low, high = (int(value, 16) for value in line.split()[0].split("-"))
    if low <= address < high:
        print(line)
        print(f"mapping_offset={address - low:#x} ({address - low} bytes)")
        break
else:
    raise SystemExit("address is not mapped")
PY
```

### Filesystem extent fragmentation

The repository includes:

```bash
contrib/swiftsync_fragmentation.py
```

It automatically finds a single running daemon. Explicit alternatives:

```bash
contrib/swiftsync_fragmentation.py --pid "$PID"
```

```bash
contrib/swiftsync_fragmentation.py \
  --directory ~/.floresta/.swiftsync-utxos-SESSION
```

It reports:

- Logical and allocated GiB.
- Physical extent count.
- Extents per GiB.
- Average extent size.

This measures filesystem fragmentation only. It does not measure partially live allocator blocks or free-list fragmentation.

For the complete FIEMAP output:

```bash
filefrag -v ~/.floresta/.swiftsync-utxos-SESSION/body
filefrag -v ~/.floresta/.swiftsync-utxos-SESSION/blobs
```

### Runtime SwiftSync logs

Capture at least these fields per height band:

- `window_mib`
- `pending_mib`
- `pending_blocks`
- `estimated_block_bytes`
- `processed_blocks`
- `processed_block_mbps`
- `queued_blocks`
- `busy_workers`
- `avg_worker_turnaround_ms`
- `avg_prevout_fetch_ms` (the combined `batch_pop`, reclamation, and decode time)
- `avg_prevout_delete_ms` (retained for compatibility and expected to remain zero)
- `avg_consensus_validation_ms`
- `dynamic_timeout_ms`
- `average_peer_ms`
- `oldest_request_age_secs`
- `major faults/sec` from `perf stat`

## Experiment protocol

For useful A/B comparisons:

1. Use the same machine, filesystem, network, hints file, and peer policy.
2. Start from a fresh temporary SwiftSync database for each run.
3. Compare the same height ranges; Bitcoin workload changes substantially by height.
4. Record the exact build ID and configuration.
5. Do not compile, link, copy large files, or run filesystem benchmarks during the measurement.
6. Separate cold-cache startup from warmed steady-state results.
7. Run each candidate more than once.
8. Keep raw logs and timestamped perf captures.

Suggested experiment matrix:

| Dimension | Candidates |
|---|---|
| SwiftSync workers | 8, 12, 16, 32 |
| Allocator block size | 1 MiB, 512 KiB, 256 KiB, 64 KiB |
| Download window | 256 MiB, 512 MiB, 1 GiB, adaptive |
| Value layout | External blobs, 32-byte inline, mixed inline/blob |
| Reclamation | Whole block, per-slot hazard pointers, epochs |
| Prevout batching | Per block, multi-block global batch |
| Frontier retry | Dynamic timeout, reserved slot, redundant request |

Primary success criteria:

- Blocks validated per second at the same height range.
- Lower major-fault rate.
- Lower prevout-fetch latency without reducing script-validation CPU utilization.
- No long intervals with `busy_workers=0` and a large buffered backlog.
- Lower body/blob high-water sizes.
- Lower internal dead space and acceptable extents/GiB.
- No database corruption or consensus divergence.
