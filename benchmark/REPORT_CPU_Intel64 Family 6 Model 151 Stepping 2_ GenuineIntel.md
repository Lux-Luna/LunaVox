# LunaVox Benchmark Report (CPU)

- **Device**: Intel64 Family 6 Model 151 Stepping 2_ GenuineIntel
- **Timestamp**: 2025-12-31 09:11:52
- **Rounds**: 1
- **Warmup**: 1

## Summary Table

| Version | Lang | Latency(avg) | RTF(avg) | RAM(avg) | RAM(peak) | VRAM(peak) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| v2 | ZH | 1702.4 ms | 0.4343 | 5407.5 MB | 5407.5 MB | N/A |
| v2 | EN | 1437.0 ms | 0.4666 | 5622.0 MB | 5622.0 MB | N/A |
| v2 | JA | 709.9 ms | 0.2958 | 5635.2 MB | 5635.2 MB | N/A |
| v2pp | ZH | 2068.1 ms | 0.5276 | 6744.6 MB | 6744.6 MB | N/A |
| v2pp | EN | 1605.6 ms | 0.5213 | 6781.3 MB | 6781.3 MB | N/A |
| v2pp | JA | 885.1 ms | 0.3688 | 6824.8 MB | 6824.8 MB | N/A |

## Pipeline Component Latency (avg)

| Version | Lang | Frontend | T2S | VITS | Vocoder Kernel |
| :--- | :--- | :--- | :--- | :--- | :--- |
| v2 | ZH | 0.00 ms | 1201.73 ms | 398.48 ms | 396.57 ms |
| v2 | EN | 0.00 ms | 1097.25 ms | 334.32 ms | 332.61 ms |
| v2 | JA | 0.00 ms | 478.20 ms | 186.46 ms | 184.37 ms |
| v2pp | ZH | 0.00 ms | 1263.28 ms | 712.91 ms | 710.22 ms |
| v2pp | EN | 0.00 ms | 1012.23 ms | 587.18 ms | 584.26 ms |
| v2pp | JA | 0.00 ms | 513.71 ms | 251.00 ms | 249.15 ms |