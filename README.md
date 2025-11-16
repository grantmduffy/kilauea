# Kilauea

Kilauea is the pythonic Vulkan render engine! We made high-performance, modern GPU rendering accessible through a simple Python interface.

## Why Kilauea?

Python is known for its simplicity and ease of use, but it's not traditionally associated with high-performance graphics. Kilauea changes that by bridging the gap between Python's accessibility and the raw power of modern GPUs.

Kilauea is built on **Vulkan**, a modern, explicit graphics and compute API that provides direct control over the GPU. By leveraging a GPU-driven rendering pipeline, Kilauea offloads the heavy lifting from the CPU. This means that the traditional performance bottlenecks of Python's interpreter are largely bypassed for rendering tasks.

Furthermore, Kilauea integrates seamlessly with optimized third-party libraries like **NumPy** for efficient data handling. All the complex vertex data and transformations are managed in highly optimized buffers that are sent directly to the GPU.

The result is a library that enables high-performance graphics projects in a very pythonic and approachable way. You get the best of both worlds: the development speed of Python and the rendering performance of a low-level API like Vulkan.

## Getting Started

Getting started with Kilauea is simple.

### 1. Installation

First, clone the repository. To ensure all dependencies are correctly installed and the package is available to your scripts, install it in "editable" mode using pip from the root of the project directory:

```bash
pip install -e .
```

This will install all the necessary dependencies, including `vulkan`, `numpy`, `glfw`, and `PyGLM`.

### 2. Your First Triangle

The best way to start is by looking at the `hello_world` example. You can run it from the project's root directory:

```bash
python -m kilauea.examples.hello_world.hello_world
```

This will open a window with an animated, colorful triangle, demonstrating the basic components of a Kilauea application:

-   **App**: The main application class that handles the window, Vulkan initialization, and the main loop.
-   **Drawable**: An object that can be drawn, containing vertices, shaders, and other rendering information.
-   **Pass**: A render pass that defines a sequence of drawing operations.
-   **Uniforms and DescriptorSets**: How you pass data (like transformations or time) from your Python script to your shaders on the GPU.

By exploring this example, you can see how these components work together to create a complete rendering application.
