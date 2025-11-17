kilauea.buffer
==============

.. automodule:: kilauea.buffer

.. autoclass:: Buffer
   :members:
   :inherited-members:

   A Vulkan buffer abstraction for GPU memory management.

   Buffer provides a high-level interface for creating Vulkan buffers and managing
   their GPU memory. It automatically handles memory allocation, type selection,
   and data transfer operations.

   **Key Features:**

   * Automatic memory type selection for optimal performance
   * Host-visible memory mapping for data uploads
   * Buffer creation for various usage patterns (vertex, index, uniform, etc.)

   **Usage Patterns:**

   * ``vertex`` - Vertex buffer for geometry data
   * ``index`` - Index buffer for indexed drawing
   * ``uniform`` - Uniform buffer (though UniformBuffer is usually preferred)

   **Usage Example:**

   .. code-block:: python

      from kilauea.buffer import Buffer
      import numpy as np

      # Create vertex buffer
      vertices = np.array([
          [-0.5, -0.5, 0.0],
          [ 0.5, -0.5, 0.0],
          [ 0.0,  0.5, 0.0]
      ], dtype=np.float32)

      vertex_buffer = Buffer(app, vertices, usage='vertex')

      # Buffer is automatically written during creation with write()

   .. automethod:: __init__(app, data: np.ndarray=None, usage='vertex')

      :param app: The Kilauea application instance
      :type app: App
      :param data: Buffer data as numpy array
      :type data: np.ndarray or None
      :param usage: Buffer usage pattern ('vertex', 'index', etc.)
      :type usage: str

   .. automethod:: write(data)

      Uploads data to the GPU buffer.

      :param data: Data to upload
      :type data: np.ndarray

.. module:: kilauea.command_buffer

.. autoclass:: CommandBuffer
   :members:
   :inherited-members:

   A Vulkan command buffer abstraction with context manager support.

   CommandBuffer provides a clean interface for recording Vulkan commands using
   Python's context manager protocol. It handles command buffer allocation,
   beginning, and ending automatically.

   **Key Features:**

   * Context manager interface for RAII-style command recording
   * Automatic command buffer lifecycle management
   * Static factory method for creating multiple buffers

   **Usage Example:**

   .. code-block:: python

      from kilauea.command_buffer import CommandBuffer

      # Create command buffers (typically done by higher-level classes)
      command_buffers = CommandBuffer.make_command_buffers(app, 3)

      # Record commands using context manager
      with command_buffers[0]:
          # Record Vulkan commands here
          vk.vkCmdDraw(command_buffer._vk_command_buffer, 3, 1, 0, 0)

   .. automethod:: make_command_buffers(app, n)

      Factory method to create multiple command buffers.

      :param app: The Kilauea application instance
      :type app: App
      :param n: Number of command buffers to create
      :type n: int
      :return: List of CommandBuffer objects
      :rtype: list

   .. automethod:: __init__(app, command_buffer)

      :param app: The Kilauea application instance
      :type app: App
      :param command_buffer: Raw Vulkan command buffer handle
      :type command_buffer: VkCommandBuffer

   .. automethod:: __enter__()

      Begins command buffer recording.

   .. automethod:: __exit__(exc_type=None, exc_value=None, traceback=None)

      Ends command buffer recording.
