from pyrite import App, UniformBuffer, Uniform, Pass, Image, Texture, Swapchain, Drawable, Compute, ComputePass
import numpy as np
import glfw
import vulkan as vk
from pathlib import Path


path = Path(__file__).parent


class MyApp(App):

    def __init__(self):
        super().__init__('MyApp', n_images=10, n_frames=9)

        triangle = 0.5 * np.array([
            [-1, -1, 0.0, 0.0],
            [1, -1, 1.0, 0.0],
            [0, 1, 0.5, 1.0]
        ], dtype=np.float32)

        # Create uniform buffer with proper initialization
        self.uniforms = UniformBuffer(self)
        self.uniforms['camera'] = Uniform(np.eye(4, dtype=np.float32), glsl_type='mat4')
        self.uniforms['time'] = Uniform(np.array([0.0], dtype=np.float32), glsl_type='float')
        self.uniforms.create()

        # Create images and textures
        self.noise_image = Image(self, format=vk.VK_FORMAT_R8G8B8A8_UNORM, width=256, height=256)
        self.noise_texture = Texture(self, self.noise_image)
        self.storage_texture = Texture(self, self.noise_image, storage=True)

        # Create compute pass and compute shader - much cleaner!
        self.compute_pass = ComputePass(self)
        self.noise_compute = Compute(
            self, path / 'noise.comp',
            uniforms=self.uniforms,
            storage_images=[self.storage_texture]
        )

        # Create graphics pass and drawable - also much cleaner!
        self.pass1 = Pass(self, clear_color=(0, 0, 0, 0))
        self.mesh1 = Drawable(
            self, triangle, 
            path / 'triangle.vert', path / 'triangle.frag', 
            self.pass1, 
            vertex_attributes=['vec2', 'vec2'],
            uniforms=self.uniforms,
            textures=[self.noise_texture]
        )
        
        self.last_time = glfw.get_time()
        self.last_count = 0
        self.fps_interval = 0.2

    def draw(self, command_buffer, swapchain_image):
        with self.compute_pass.start(command_buffer, self.noise_image) as pass_ctx:
            self.noise_compute.dispatch(command_buffer, 256//8, 256//8, 1)
        with self.pass1.start(command_buffer, swapchain_image) as pass_ctx:
            self.mesh1.draw(command_buffer, 0)
        
    def main_loop(self):
        # handle user input, update uniforms, etc.
        t = glfw.get_time()
        
        # Test uniform updates - this demonstrates the persistent mapping working!
        # Update time uniform
        self.uniforms['time'][0] = t
        
        # Add circular translation
        radius = 0.2
        self.uniforms['camera'][3, 0] = radius * np.sin(t)  # X translation
        self.uniforms['camera'][3, 1] = radius * np.cos(t)  # Y translation
        
        # FPS display
        if t - self.last_time > self.fps_interval:
            fps = (self.frame_count - self.last_count) / (t - self.last_time)
            self.last_count = self.frame_count
            self.last_time = t
            glfw.set_window_title(self.window, f'{self.title} {fps:.2f}fps - Time: {t:.1f}')


if __name__ == '__main__':
    app = MyApp()
    app.run()
