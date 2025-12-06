## Sum Reduce on Tensor Core

```
$(NVCC) -o ./bin/wmma_reduce reduction_wmma.cu -arch=sm_70 -lcudart
```

```
RTX 3060 - mem bandwidth = 360 GB/s

Tensor Core Reduction with N = 8388608
		TCU Reduce		CUB
Time:		0.174080 ms		0.157600 ms
Throughput:	96.376472 GB/s		106.454414 GB/s
Results:	8388608.000000 	8388608.000000 (expected 8388608.000000)

Tensor Core Reduction with N = 67108864
		TCU Reduce		CUB
Time:		0.728128 ms		1.301600 ms
Throughput:	184.332596 GB/s		103.117493 GB/s
Results:	67108864.000000 	67108864.000000 (expected 67108864.000000

Tensor Core Reduction with N = 536870912
		TCU Reduce		CUB
Time:		3.931136 ms		9.208832 ms
Throughput:	273.137817 GB/s		116.599136 GB/s
Results:	536870912.000000 	536870912.000000 (expected 536870912.000000)
```
