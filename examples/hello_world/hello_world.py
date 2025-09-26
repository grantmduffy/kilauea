from pyrite import App, UniformBuffer, Uniform, Pass, Image, Texture, Swapchain, Drawable
import numpy as np
import glfw
import vulkan as vk
from pathlib import Path


path = Path(__file__).parent


class MyApp(App):

    def __init__(self):
        super().__init__('MyApp', n_images=10, n_frames=9)

        triangle = 0.5 * np.array([
            [-1, -1, 1.0, 0.0, 0.0],
            [1, -1, 0.0, 1.0, 0.0],
            [0, 1, 0.0, 0.0, 1.0]
        ], dtype=np.float32)

        # Create uniform buffer with proper initialization
        self.uniforms = UniformBuffer(self)
        self.uniforms['camera'] = Uniform(np.eye(4, dtype=np.float32), glsl_type='mat4')
        self.uniforms['time'] = Uniform(np.array([0.0], dtype=np.float32), glsl_type='float')
        self.uniforms.create()

        self.pass1 = Pass(self, clear_color=(0, 0, 0, 0))
        self.mesh1 = Drawable(
            self, triangle, 
            path / 'triangle.vert', path / 'triangle.frag', 
            self.pass1, 
            vertex_attributes=['vec2', 'vec3'],
            uniforms=self.uniforms
        )
        
        self.last_time = glfw.get_time()
        self.last_count = 0
        self.fps_interval = 0.2

    def draw(self, command_buffer, swapchain_image):
        with self.pass1.start(command_buffer, swapchain_image) as pass_ctx:
            # Set viewport and scissor for the render pass
            viewport = vk.VkViewport(
                x=0, y=0, 
                width=float(pass_ctx.render_extent.width), 
                height=float(pass_ctx.render_extent.height),
                minDepth=0.0, maxDepth=1.0
            )
            scissor = vk.VkRect2D(offset=[0, 0], extent=pass_ctx.render_extent)
            
            vk.vkCmdSetViewport(command_buffer._vk_command_buffer, 0, 1, [viewport])
            vk.vkCmdSetScissor(command_buffer._vk_command_buffer, 0, 1, [scissor])
            
            self.mesh1.draw(command_buffer, 0)
        
    def main_loop(self):
        # handle user input, update uniforms, etc.
        t = glfw.get_time()
        
        # Test uniform updates - this demonstrates the persistent mapping working!
        # Update time uniform
        self.uniforms['time'][0] = t
        
        # Animate camera matrix - translate X based on time
        self.uniforms['camera'][3, 0] = 0.2 * np.sin(t)  # X translation
        self.uniforms['camera'][3, 1] = 0.2 * np.cos(t)
        
        # FPS display
        if t - self.last_time > self.fps_interval:
            fps = (self.frame_count - self.last_count) / (t - self.last_time)
            self.last_count = self.frame_count
            self.last_time = t
            glfw.set_window_title(self.window, f'{self.title} {fps:.2f}fps - Time: {t:.1f}')


if __name__ == '__main__':
    app = MyApp()
    app.run()
