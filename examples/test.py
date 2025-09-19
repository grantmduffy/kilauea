from pyrite import App, UniformBuffer, Uniform, Pass, Image, Texture, Swapchain, Drawable
import numpy as np
import glfw
import vulkan as vk


class MyApp(App):

    def __init__(self):
        super().__init__('MyApp')

        triangle = np.array([
            [-0.8, -0.8, 1.0, 0.0, 0.0],
            [0.8, -0.8, 0.0, 1.0, 0.0],
            [0.0, 0.8, 0.0, 0.0, 1.0]
        ], dtype=np.float32)

        full_screen = np.array([
            [-1, -1], [1, -1], [1, 1],
            [-1, -1], [1, 1], [-1, 1]
        ], dtype=np.float32)

        # Create uniform buffer with proper initialization
        self.uniforms = UniformBuffer(self)
        self.uniforms['camera'] = Uniform(np.eye(4, dtype=np.float32), glsl_type='mat4')
        self.uniforms['time'] = Uniform(np.array([0.0], dtype=np.float32), glsl_type='float')
        self.uniforms.make_buffer()

        self.pass1 = Pass(self, final_layout=vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
        self.pass2 = Pass(self)
        self.deferred_image = Image(self, render_pass=self.pass1)
        self.deferred_texture = Texture(self, self.deferred_image)
        
        self.mesh1 = Drawable(
            self, triangle, 
            './glsl/triangle.vert', './glsl/triangle.frag', 
            self.pass1, 
            vertex_attributes=['vec2', 'vec3'],
            uniforms=self.uniforms
        )
        self.blur = Drawable(
            self, full_screen,
            './glsl/blur.vert', './glsl/blur.frag',
            self.pass2,
            vertex_attributes=['vec2'],
            uniforms=self.uniforms,
            textures=[self.deferred_texture]
        )
        
        self.last_time = glfw.get_time()
        self.last_count = 0
        self.fps_interval = 0.2

    def draw(self, command_buffer, swapchain_image):
        with self.pass1.start(command_buffer, self.deferred_image):
            self.mesh1.draw(command_buffer, 0)

        # Transition the off-screen image for shader reading
        self.deferred_image.transition_layout(command_buffer, vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)

        with self.pass2.start(command_buffer, swapchain_image):
            self.blur.draw(command_buffer, 0)

    def cleanup_swapchain(self):
        self.mesh1.destroy()
        self.pass1.destroy()
        self.swapchain.destroy()

    def create_swapchain(self):
        # Store the old n_images value
        old_n_images = getattr(self.swapchain, 'n_images', 4)
        
        self.swapchain = Swapchain(self, old_n_images)
        self.pass1 = Pass(self)
        self.mesh1.render_pass = self.pass1
        self.mesh1.create_pipeline()

    def main_loop(self):
        # handle user input, update uniforms, etc.
        t = glfw.get_time()
        
        # Test uniform updates - this demonstrates the persistent mapping working!
        # Update time uniform
        self.uniforms['time'][0] = t
        
        # Animate camera matrix - translate X based on time
        self.uniforms['camera'][0, 3] = 0.2 * np.sin(t)  # X translation
        
        # FPS display
        if t - self.last_time > self.fps_interval:
            fps = (self.frame_count - self.last_count) / (t - self.last_time)
            self.last_count = self.frame_count
            self.last_time = t
            glfw.set_window_title(self.window, f'{self.title} {fps:.2f}fps - Time: {t:.1f}')


if __name__ == '__main__':
    app = MyApp()
    app.run()
