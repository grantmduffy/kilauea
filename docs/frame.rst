kilauea.frame
=============

.. automodule:: kilauea.frame

.. autoclass:: Frame
   :members:
   :inherited-members:

   A frame abstraction that manages synchronization objects and command submission timing.

   Frame coordinates the complex synchronization between CPU and GPU operations across
   multiple frames in flight. It manages semaphores, fences, and their relationships
   between different rendering passes.

   **Key Responsibilities:**

   * Fence management for CPU-GPU synchronization
   * Semaphore coordination between render passes
   * Frame index tracking and rotation

   .. automethod:: __init__(parent)

      :param parent: The parent App instance
      :type parent: App

   .. automethod:: get_semaphores_signaled_by(o)

      Returns semaphores signaled by the given object.

   .. automethod:: get_semaphores_consumed_by(o)

      Returns semaphores consumed by the given object.

.. autoclass:: Semaphore
   :members:
   :inherited-members:

   A Vulkan semaphore for GPU-GPU synchronization between queue operations.

   Semaphores provide synchronization between different GPU operations, ensuring
   that operations complete in the correct order. They are used to coordinate
   between render passes and with the swapchain.

   .. automethod:: __init__(app)

      :param app: The Kilauea application instance
      :type app: App

   .. automethod:: destroy()

      Cleans up the semaphore resource.

.. autoclass:: Fence
   :members:
   :inherited-members:

   A Vulkan fence for CPU-GPU synchronization.

   Fences allow the CPU to wait for GPU operations to complete, ensuring
   proper synchronization when the CPU needs to access resources that may
   still be in use by the GPU.

   .. automethod:: __init__(app, timeout=1000000000)

      :param app: The Kilauea application instance
      :type app: App
      :param timeout: Timeout in nanoseconds for wait operations
      :type timeout: int

   .. automethod:: wait()

      Waits for the fence to be signaled by the GPU.

   .. automethod:: reset()

      Resets the fence to unsignaled state.

.. autoclass:: Image
   :members:
   :inherited-members:

   A Vulkan image abstraction with layout management and view creation.

   Image handles the creation and management of Vulkan images, including
   automatic view creation and layout transitions. It supports various
   image formats and usage patterns.

   **Key Features:**

   * Automatic image view creation
   * Layout transition management
   * Memory allocation and binding

   .. automethod:: __init__(app, width, height, format, usage, initial_layout=vk.VK_IMAGE_LAYOUT_UNDEFINED, mip_levels=1, samples=vk.VK_SAMPLE_COUNT_1_BIT)

      :param app: The Kilauea application instance
      :type app: App
      :param width: Image width in pixels
      :type width: int
      :param height: Image height in pixels
      :type height: int
      :param format: Vulkan image format
      :type format: VkFormat
      :param usage: Image usage flags (color attachment, sampled, etc.)
      :type usage: VkImageUsageFlags
      :param initial_layout: Initial image layout
      :type initial_layout: VkImageLayout
      :param mip_levels: Number of mip levels
      :type mip_levels: int
      :param samples: Sample count for multisampling
      :type samples: VkSampleCountFlagBits

   .. automethod:: transition_layout(command_buffer, new_layout, image_i=0)

      Records a layout transition command in the given command buffer.

   .. automethod:: destroy()

      Cleans up image resources.

.. autoclass:: Texture
   :members:
   :inherited-members:

   A texture resource that combines an Image with sampler and descriptor management.

   Texture extends Image with additional functionality for use as a sampled texture
   in shaders, including sampler creation and descriptor set integration.

   **Key Features:**

   * Automatic sampler creation
   * Descriptor set integration
   * Support for storage textures (compute shader writable)

   **Usage Example:**

   .. code-block:: python

      from kilauea.frame import Texture

      # Create a sampled texture
      texture = Texture(app, image_data)

      # Add to descriptor set for shader access
      descriptor_set.add(texture, stages=vk.VK_SHADER_STAGE_FRAGMENT_BIT)

   .. automethod:: __init__(app, image, storage=False)

      :param app: The Kilauea application instance
      :type app: App
      :param image: The underlying Image object or raw image data
      :type image: Image or np.ndarray
      :param storage: Whether this is a storage texture (compute shader writable)
      :type storage: bool

   .. automethod:: get_layout_binding(binding: int, stages)

      Returns descriptor set layout binding for this texture.

   .. automethod:: get_write_descriptor(_vk_descriptor_set, i_image: int, i_binding: int)

      Returns descriptor write for updating descriptor sets.

   .. automethod:: destroy()

      Cleans up texture resources.
