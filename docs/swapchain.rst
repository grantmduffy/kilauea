kilauea.swapchain
==================

.. automodule:: kilauea.swapchain

.. autoclass:: Swapchain
   :members:
   :inherited-members:

   Swapchain management for window presentation and multi-buffered rendering.

   The Swapchain class handles Vulkan swapchain creation, image acquisition, and
   presentation. It manages multiple swapchain images for smooth rendering and
   proper synchronization with the display.

   **Key Features:**

   * Automatic swapchain recreation on window resize
   * Image acquisition and presentation management
   * Synchronization with rendering passes
   * Surface format and present mode negotiation

   **Usage Example:**

   .. code-block:: python

      # Swapchain is automatically created by App
      # Access via app.swapchain

      def draw(self, image_i):
          # Get next available image
          image_i = self.swapchain.get_next_image(frame)

          # Render to swapchain image...

          # Present when done
          self.swapchain.present_image(image_i, presentation_semaphores)

   .. automethod:: __init__(app, n_images, composite_alpha='opaque', wait_for=None, extent=None)

      :param app: The Kilauea application instance
      :type app: App
      :param n_images: Number of swapchain images to request
      :type n_images: int
      :param composite_alpha: Compositing mode ('opaque', 'pre_multiplied', etc.)
      :type composite_alpha: str
      :param wait_for: Objects to wait for before presentation
      :type wait_for: list or None
      :param extent: Optional surface extent override
      :type extent: tuple or None

   .. automethod:: get_next_image(frame, timeout=1000000000)

      Acquires the next available swapchain image for rendering.

      :param frame: Current frame object for synchronization
      :type frame: Frame
      :param timeout: Timeout in nanoseconds
      :type timeout: int
      :return: Index of the acquired image
      :rtype: int

   .. automethod:: present_image(image_i, presentation_semaphores)

      Presents a rendered image to the display.

      :param image_i: Index of the image to present
      :type image_i: int
      :param presentation_semaphores: Semaphores to wait for before presenting
      :type presentation_semaphores: list

   .. automethod:: present_wait_for(passes)

      Sets which render passes to wait for before presentation.

      :param passes: Render passes that must complete before presentation
      :type passes: list of Pass

   .. automethod:: create(width, height)

      Creates or recreates the swapchain with the given dimensions.

      :param width: Swapchain width in pixels
      :type width: int
      :param height: Swapchain height in pixels
      :type height: int

   .. automethod:: destroy()

      Cleans up swapchain resources.
