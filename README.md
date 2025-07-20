# PSI InSAR in Rust

This is a WIP library to perform Permanent Scatterer Interferometry and Interferometric SAR workloads with Sentinel-1 SLC data.

Currently, the coregistration implementation is based on the paper [^1].

## References

- https://forum.step.esa.int/t/insar-dinsar-perpendicular-baseline-calculation/3776/42
- Java code from SNAP: [InSARStackOverview.java#L292-20](https://github.com/senbox-org/s1tbx/blob/eedb363807ffb78a782dbbb23f10907e488d06cb/s1tbx-op-insar/src/main/java/org/esa/s1tbx/insar/gpf/InSARStackOverview.java#L292)
[^1]: Imperatore, P.; Sansosti, E. Multithreading Based Parallel Processing for Image Geometric Coregistration in SAR Interferometry. Remote Sens. 2021, 13, 1963. https://doi.org/10.3390/rs13101963
