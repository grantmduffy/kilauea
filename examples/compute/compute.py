from kilauea import App, UniformBuffer, Uniform, Pass, Image, Texture, Swapchain, Drawable, Compute, DescriptorSet
import numpy as np
import glfw
import vulkan as vk
from pathlib import Path


path = Path(__file__).parent


class MyApp(App):

    def __init__(self):
        super().__init__('MyApp', n_images=4, n_frames=3)

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
        self.noise_image = Image(self, format=vk.VK_FORMAT_R8G8B8A8_UNORM, width=256, height=256, n_images=self.swapchain.n_images)
        self.noise_texture = Texture(self, self.noise_image)
        self.storage_texture = Texture(self, self.noise_image, storage=True)

        # create descriptor sets
        ds1 = DescriptorSet(self, n_images=self.swapchain.n_images)
        ds1.add(self.uniforms, stages=vk.VK_SHADER_STAGE_COMPUTE_BIT)
        ds1.add(self.storage_texture, stages=vk.VK_SHADER_STAGE_COMPUTE_BIT)
        ds1.create()
        ds2 = DescriptorSet(self, n_images=self.swapchain.n_images)
        ds2.add(self.uniforms, stages=vk.VK_SHADER_STAGE_ALL_GRAPHICS)
        ds2.add(self.noise_texture, stages=vk.VK_SHADER_STAGE_FRAGMENT_BIT)
        ds2.create()
        
        # Create compute pass and compute shader
        self.compute_pass = Pass(self, wait_for=[self.swapchain], compute=True)
        self.noise_compute = Compute(
            self, path / 'noise.comp',
            descriptor_sets=[ds1]
        )

        # Create graphics pass and drawable
        self.pass1 = Pass(self, clear_color=(0, 0, 0, 0), wait_for=[self.compute_pass])
        self.mesh1 = Drawable(
            self, triangle, 
            path / 'triangle.vert', path / 'triangle.frag', 
            self.pass1, 
            vertex_attributes=['vec2', 'vec2'],
            descriptor_sets=[ds2]
        )
        self.swapchain.present_wait_for([self.pass1])
        
        self.last_time = glfw.get_time()
        self.last_count = 0
        self.fps_interval = 0.2

    def draw(self, image_i):
        # TODO maybe have PassContext.__enter__ setup a global current_command_buffer that .dispatch and .draw can use? 
        with self.compute_pass.start(image_i, self.noise_image) as pass_ctx:
            self.noise_compute.dispatch(pass_ctx, 256//8, 256//8, 1)
        with self.pass1.start(image_i, self.swapchain.images) as pass_ctx:
            self.mesh1.draw(pass_ctx)
        
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
