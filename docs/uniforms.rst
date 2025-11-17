kilauea.uniforms
================

.. automodule:: kilauea.uniforms

.. autoclass:: Uniform
   :members:
   :inherited-members:

   A single uniform variable that can be mapped to GPU memory for persistent access.

   The ``Uniform`` class handles the complex alignment requirements of GLSL's std140 layout rules,
   ensuring proper memory layout when uniforms are grouped into a UniformBuffer.

   **Key Features:**

   * Automatic GLSL std140 alignment calculations
   * Persistent memory mapping for direct GPU access
   * Type-aware size and alignment handling for float, vec3, mat4, etc.

   **Usage Example:**

   .. code-block:: python

      from kilauea.uniforms import Uniform
      import numpy as np

      # Create typed uniforms
      time_uniform = Uniform(np.array([0.0], dtype=np.float32), glsl_type='float')
      camera_matrix = Uniform(np.eye(4, dtype=np.float32), glsl_type='mat4')
      color_uniform = Uniform(np.array([1.0, 0.5, 0.0], dtype=np.float32), glsl_type='vec3')

   .. automethod:: __init__(data=None, glsl_type=None)

      :param data: The uniform data as a numpy array
      :type data: np.ndarray or None
      :param glsl_type: The GLSL type string ('float', 'vec3', 'mat4', etc.)
      :type glsl_type: str or None

   .. automethod:: __getitem__(k)

      Access uniform data elements (returns mapped GPU memory when available).

   .. automethod:: __setitem__(k, v)

      Set uniform data elements (writes to GPU memory when mapped).

   .. automethod:: map_to_location(mapped_memory, offset)

      Maps this uniform to a location in persistently mapped GPU memory.

.. autoclass:: UniformBuffer
   :members:
   :inherited-members:

   A collection of uniforms stored in a single Vulkan uniform buffer with persistent mapping.

   UniformBuffer automatically handles the complexity of GLSL std140 layout rules, ensuring
   that uniform data is properly aligned and accessible from shaders. The buffer uses
   persistent mapping for efficient CPU-GPU data transfer.

   **Key Features:**

   * Automatic std140 layout with proper alignment calculations
   * Persistent buffer mapping for zero-copy uniform updates
   * Descriptor set integration for shader access
   * Memory type selection for optimal performance

   **Usage Example:**

   .. code-block:: python

      from kilauea.uniforms import UniformBuffer, Uniform
      import numpy as np

      # Create uniform buffer
      uniforms = UniformBuffer(app)

      # Add typed uniforms
      uniforms['camera'] = Uniform(np.eye(4, dtype=np.float32), glsl_type='mat4')
      uniforms['time'] = Uniform(np.array([0.0], dtype=np.float32), glsl_type='float')
      uniforms['color'] = Uniform(np.array([1.0, 0.0, 0.0], dtype=np.float32), glsl_type='vec3')

      # Create the buffer (calculates layout and maps memory)
      uniforms.create()

      # Update uniforms (direct GPU memory access)
      uniforms['time'][0] = glfw.get_time()
      uniforms['camera'][3, 0] = np.sin(time)  # Translation

   .. automethod:: __init__(app, data=None)

      :param app: The Kilauea application instance
      :type app: App
      :param data: Optional initial uniform data
      :type data: dict or None

   .. automethod:: __setitem__(uniform_name, uniform)

      Add a uniform to the buffer.

   .. automethod:: __getitem__(uniform_name)

      Retrieve a uniform from the buffer.

   .. automethod:: get_layout_binding(binding: int, stages)

      Returns a VkDescriptorSetLayoutBinding for this uniform buffer.

   .. automethod:: get_write_descriptor(_vk_descriptor_set, i_image: int, i_binding: int)

      Returns a VkWriteDescriptorSet for updating descriptor sets.

   .. automethod:: create()

      Creates the Vulkan buffer and maps all uniforms to GPU memory. Must be called
      after all uniforms are added but before rendering.

   .. automethod:: destroy()

      Cleans up Vulkan resources.
