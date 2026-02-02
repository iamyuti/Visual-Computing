# Python Visual Computing

A collection of visual computing projects demonstrating image processing and computer graphics fundamentals, implemented from scratch in Python.

## Projects

### [01 – RAW Image Processing](01-Raw-Image-Processing/)

A complete ISP (Image Signal Processing) pipeline for converting RAW camera images into high-quality RGB images.

**Key Features:**
- Black level correction & sensor normalization
- Bayer demosaicing (bilinear interpolation)
- Automatic & manual white balance
- Gamma correction (sRGB)
- Histogram stretch & binary thresholding

![RAW Processing Result](01-Raw-Image-Processing/results/IMG_0_processed.png)

---

### [02 – 3D Software Rasterizer](02-3D-Software-Renderer/)

A from-scratch software rasterizer demonstrating core computer graphics algorithms without GPU acceleration.

**Key Features:**
- Sutherland-Hodgman polygon clipping (NDC volume)
- DDA line rasterization (wireframe)
- Barycentric fill rasterization (solid triangles)
- Z-buffer depth testing
- PLY model loading

![Torus Render](02-3D-Software-Renderer/examples/torus_fill.png)

---

## Tech Stack

- **Python 3.8+**
- **NumPy** – Numerical operations
- **Pillow** – Image I/O
- **SciPy** – Interpolation (RAW pipeline)

## About

These projects focus on implementing fundamental visual computing algorithms from scratch. External libraries are only used for I/O and basic numerical operations — all core algorithms (clipping, rasterization, demosaicing, etc.) are implemented manually. The goal was to deeply understand how image processing pipelines and 3D rendering engines work under the hood.

## License

MIT License
