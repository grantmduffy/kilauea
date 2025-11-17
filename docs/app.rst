kilauea.app
===========

.. automodule:: kilauea.app

.. autoclass:: App
   :members:
   :inherited-members:

   The main application class for Kilauea. This class handles window creation,
   Vulkan initialization, device selection, and the main render loop.

   **Key Responsibilities:**

   * Initializes GLFW and creates a window
   * Sets up Vulkan instance, physical and logical devices
   * Manages the swapchain and framebuffers
   * Provides the main render loop
   * Handles synchronization between CPU and GPU

   **Usage Example:**

   .. code-block:: python

      from kilauea.app import App

      class MyApp(App):
          def __init__(self):
              super().__init__(title='My App', size=(800, 600))

          def draw(self, image_i):
              # Your render commands here
              pass

      if __name__ == '__main__':
          app = MyApp()
          app.run()

   .. automethod:: __init__(title='Kilauea', size=(640, 480), n_frames=3, n_images=4, version=(1, 3, 0), engine_name='Kilauea', device_preference=['discrete_gpu', 'integrated_gpu', 'virtual_gpu', 'cpu'], surface_format=<vulkan_format>, color_space=<vulkan_color_space>)

      Initializes the Kilauea application.

      :param title: The title of the application window
      :type title: str
      :param size: The initial window size in pixels, or 'fullscreen' for fullscreen mode
      :type size: tuple(int, int) or str
      :param n_frames: Number of frames in flight (controls buffering)
      :type n_frames: int
      :param n_images: Number of swapchain images
      :type n_images: int
      :param version: Vulkan API version to request
      :type version: tuple(int, int, int)
      :param engine_name: Name of the engine for Vulkan metadata
      :type engine_name: str
      :param device_preference: Preferred order of GPU device types
      :type device_preference: list[str]
      :param surface_format: Desired surface format for the swapchain
      :type surface_format: int
      :param color_space: Desired color space for the swapchain
      :type color_space: int

   .. automethod:: initialize_objects()

   .. automethod:: init_vk()

   .. automethod:: get_memory_type_index(type_bits, properties)

   .. automethod:: create_descriptor_pool(max_sets=100)

   .. automethod:: print_device_properties()

   .. automethod:: print_device_properties2()

   .. automethod:: debug_shader_spirv(shader_path, stage_name="unknown")

   .. automethod:: analyze_spirv_output(spirv_text, shader_path)

   .. automethod:: debug_pipeline_executable_properties(pipeline)

   .. automethod:: debug_pipeline_executable_statistics(stats_func, pipeline, executable_index)

   .. automethod:: debug_graphics_pipeline_creation(shader_stages, vertex_input_state, input_assembly_state, viewport_state, rasterization_state, multisample_state, color_blend_state, render_pass, pipeline_layout, subpass=0)

   .. automethod:: debug_shader_module_creation(spirv_code, stage_name="unknown")

   .. automethod:: debug_callback(*args)

   .. automethod:: clear_validation_messages()

   .. automethod:: get_recent_validation_messages(count=10)

   .. automethod:: get_next_frame()

   .. automethod:: main_loop()

   .. automethod:: record_draw_commands()

   .. automethod:: draw(command_buffer, image)

      Must be implemented by subclasses to define the rendering operations for each frame.

      :param command_buffer: The Vulkan command buffer to record commands into
      :param image: The current swapchain image index

   .. automethod:: graphics_loop()

   .. automethod:: submit_commands(image_i, frame)

   .. automethod:: run()

      Starts the main application loop. This method will block until the application is closed.

   .. automethod:: destroy()

      Cleans up Vulkan resources and terminates GLFW.

   .. automethod:: framebuffer_resize_callback(window, width, height)

      Callback for window resize events.

   .. automethod:: recreate_swapchain()

   .. automethod:: cleanup_swapchain()

   .. automethod:: create_swapchain(w, h)
