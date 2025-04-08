
If we would benchmark our implementation from the previous chapter, we will notice that starting from  the achievable FLOPS decreases starting

In our current implementation we don't take advantage of the cache system and we ineffectively move the data between the main memory and the registers. This can be especially seen by increasing the matrix size problem, as the matrices won't fit the cache.