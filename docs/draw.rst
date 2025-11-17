kilauea.draw
===========

.. automodule:: kilauea.draw

.. autoclass:: Pass
   :members:
   :inherited-members:

   A render pass abstraction that manages Vulkan render pass creation and command recording.

   The Pass class handles the complexity of Vulkan render pass setup, including attachment
   descriptions, subpass dependencies, and clear values. It provides a context manager
   interface for recording rendering commands.

   **Key Features:**

   * Automatic render pass creation with proper attachment setup
   * Support for both graphics and compute passes
   * Semaphore synchronization management between passes
   * Framebuffer creation and management

   **Usage Example:**

   .. code-block:: python

      from kilauea.draw import Pass

      # Create a graphics pass that clears to black
      render_pass = Pass(app, clear_color=(0, 0, 0, 1.0))

      # Use in draw method
      def draw(self, image_i):
          with render_pass.start(image_i, self.swapchain.images) as pass_ctx:
              # Record rendering commands here
              my_drawable.draw(pass_ctx)

   .. automethod:: __init__(app, clear_color=(0, 0, 0, 0), final_layout=vk.VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, n_images=None, wait_for=None, wait_stage=None, compute=False)

      :param app: The Kilauea application instance
      :type app: App
      :param clear_color: RGBA clear color (None disables clearing)
      :type clear_color: tuple or None
      :param final_layout: Final image layout for attachments
      :type final_layout: VkImageLayout
      :param n_images: Number of images this pass operates on
      :type n_images: int or None
      :param wait_for: List of passes to wait for before executing
      :type wait_for: list or None
      :param wait_stage: Pipeline stage to wait at
      :type wait_stage: VkPipelineStageFlags or None
      :param compute: Whether this is a compute pass (no render pass setup)
      :type compute: bool

   .. automethod:: start(image_i, target_images)

      Returns a PassContext for recording commands in this pass.

.. autoclass:: PassContext
   :members:
   :inherited-members:

   Context manager for recording rendering commands within a Pass.

   PassContext handles the setup and cleanup of Vulkan command recording,
   including render pass begin/end for graphics passes and image layout
   transitions for compute passes.

   .. automethod:: __init__(render_pass, image_i, target_images)

      :param render_pass: The parent Pass instance
      :type render_pass: Pass
      :param image_i: Current swapchain image index
      :type image_i: int
      :param target_images: Target images for rendering (usually swapchain images)
      :type target_images: list

.. autoclass:: Drawable
   :members:
   :inherited-members:

   A drawable object that manages vertex data, shaders, and pipeline state for rendering.

   Drawable combines vertex buffers, shaders, and pipeline configuration into a single
   object that can be efficiently rendered. It automatically creates and manages Vulkan
   graphics pipelines with proper vertex input layouts and descriptor sets.

   **Key Features:**

   * Automatic pipeline creation with shader compilation
   * Flexible vertex attribute mapping
   * Support for indexed and non-indexed drawing
   * Built-in descriptor set binding

   **Supported Vertex Attributes:**

   * ``vec2``: 2D vectors (8 bytes)
   * ``vec3``: 3D vectors (12 bytes, aligned to 16)
   * ``vec4``: 4D vectors (16 bytes)

   **Usage Example:**

   .. code-block:: python

      from kilauea.draw import Drawable
      import numpy as np

      # Create vertex data (position + color)
      vertices = np.array([
          [-0.5, -0.5, 1.0, 0.0, 0.0],  # pos.x, pos.y, r, g, b
          [ 0.5, -0.5, 0.0, 1.0, 0.0],
          [ 0.0,  0.5, 0.0, 0.0, 1.0]
      ], dtype=np.float32)

      # Create drawable with shaders
      triangle = Drawable(
          app, vertices,
          vertex_shader='triangle.vert',
          fragment_shader='triangle.frag',
          render_pass=my_pass,
          vertex_attributes=['vec2', 'vec3'],  # position, color
          descriptor_sets=[uniforms_descriptor]
      )

      # Draw in render pass
      def draw(self, image_i):
          with my_pass.start(image_i, self.swapchain.images) as pass_ctx:
              triangle.draw(pass_ctx)

   .. automethod:: __init__(app, vertices: np.ndarray, vertex_shader: Shader | str | Path, fragment_shader: Shader | str, render_pass, indices: np.ndarray=None, targets=None, vertex_attributes=None, descriptor_sets=None, topology='triangle_list', cull_mode='back')

      :param app: The Kilauea application instance
      :type app: App
      :param vertices: Vertex data as numpy array
      :type vertices: np.ndarray
      :param vertex_shader: Vertex shader file path or Shader object
      :type vertex_shader: Shader, str, or Path
      :param fragment_shader: Fragment shader file path or Shader object
      :type fragment_shader: Shader, str, or Path
      :param render_pass: The render pass this drawable belongs to
      :type render_pass: Pass
      :param indices: Optional index buffer data
      :type indices: np.ndarray or None
      :param vertex_attributes: List of vertex attribute types
      :type vertex_attributes: list of str
      :param descriptor_sets: Descriptor sets to bind
      :type descriptor_sets: list of DescriptorSet
      :param topology: Primitive topology ('triangle_list', 'triangle_strip', etc.)
      :type topology: str
      :param cull_mode: Face culling mode ('none', 'front', 'back')
      :type cull_mode: str

   .. automethod:: draw(pass_context: PassContext)

      Records drawing commands for this drawable in the given pass context.

.. autoclass:: Compute
   :members:
   :inherited-members:

   A compute workload that manages compute shaders and dispatch operations.

   Compute provides an abstraction for executing compute shaders on the GPU,
   handling pipeline creation and descriptor set binding for compute workloads.

   **Usage Example:**

   .. code-block:: python

      from kilauea.draw import Compute

      # Create compute workload
      compute_work = Compute(
          app,
          compute_shader='noise.comp',
          descriptor_sets=[texture_descriptors]
      )

      # Dispatch in compute pass
      def draw(self, image_i):
          with compute_pass.start(image_i, self.swapchain.images) as pass_ctx:
              compute_work.dispatch(pass_ctx, 32, 32, 1)  # 32x32 workgroups

   .. automethod:: __init__(app, compute_shader: Shader | str | Path, descriptor_sets=None)

      :param app: The Kilauea application instance
      :type app: App
      :param compute_shader: Compute shader file path or Shader object
      :type compute_shader: Shader, str, or Path
      :param descriptor_sets: Descriptor sets to bind
      :type descriptor_sets: list of DescriptorSet

   .. automethod:: dispatch(pass_context: PassContext, x, y, z)

      Dispatches the compute shader with the given workgroup dimensions.
