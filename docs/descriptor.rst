kilauea.descriptor
==================

.. automodule:: kilauea.descriptor

.. autoclass:: DescriptorSet
   :members:
   :inherited-members:

   A Vulkan descriptor set abstraction that manages descriptor set layouts and updates.

   DescriptorSet provides a high-level interface for creating and managing Vulkan descriptor
   sets, automatically handling layout creation, allocation, and updates. It supports
   binding multiple resources (uniform buffers, textures, etc.) with proper staging.

   **Key Features:**

   * Automatic descriptor set layout creation from bound resources
   * Support for multiple swapchain images with separate descriptor sets
   * Bulk descriptor updates for efficiency
   * Static binding method for use in command buffers

   **Usage Example:**

   .. code-block:: python

      from kilauea.descriptor import DescriptorSet

      # Create descriptor set for N swapchain images
      descriptor_set = DescriptorSet(app, n_images=swapchain.n_images)

      # Add resources (uniform buffers, textures, etc.)
      descriptor_set.add(uniform_buffer, stages=vk.VK_SHADER_STAGE_VERTEX_BIT)
      descriptor_set.add(texture, stages=vk.VK_SHADER_STAGE_FRAGMENT_BIT)

      # Create the descriptor sets
      descriptor_set.create()

      # Bind in command buffer
      DescriptorSet.bind(pipeline_layout, command_buffer, [descriptor_set], image_i)

   .. automethod:: __init__(app, n_images=1)

      :param app: The Kilauea application instance
      :type app: App
      :param n_images: Number of descriptor sets to create (usually matches swapchain images)
      :type n_images: int

   .. automethod:: add(item, stages=vk.VK_SHADER_STAGE_COMPUTE_BIT)

      Adds a resource (UniformBuffer, Texture, etc.) to this descriptor set.

      :param item: The resource to add (must implement get_layout_binding and get_write_descriptor)
      :type item: object
      :param stages: Shader stages where this resource will be accessed
      :type stages: VkShaderStageFlags

   .. automethod:: create(update=True)

      Creates the descriptor set layout and allocates descriptor sets.

      :param update: Whether to immediately update descriptor sets with resource bindings
      :type update: bool

   .. automethod:: update()

      Updates all descriptor sets with the current resource bindings. Called automatically
      by create() unless update=False.

   .. automethod:: bind(pipeline_layout, command_buffer, descriptor_sets, image_i: int, first_set=0, bind_point=vk.VK_PIPELINE_BIND_POINT_GRAPHICS)

      Static method to bind descriptor sets in a command buffer.

      :param pipeline_layout: The pipeline layout to bind against
      :type pipeline_layout: VkPipelineLayout
      :param command_buffer: Command buffer to record into
      :type command_buffer: CommandBuffer
      :param descriptor_sets: List of DescriptorSet objects to bind
      :type descriptor_sets: list of DescriptorSet
      :param image_i: Current swapchain image index
      :type image_i: int
      :param first_set: First descriptor set slot to bind to
      :type first_set: int
      :param bind_point: Pipeline bind point (graphics or compute)
      :type bind_point: VkPipelineBindPoint

   .. automethod:: destroy()

      Cleans up descriptor set layout resources.
