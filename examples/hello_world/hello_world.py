# To run this example, you must first install kilauea in editable mode
# from the root directory of the project (the one containing setup.py):
# pip install -e .
#
# Then, you can run this script from the same root directory using:
# python -m kilauea.examples.hello_world.hello_world

from kilauea.kilauea import App, UniformBuffer, Uniform, Pass, Drawable, DescriptorSet
import numpy as np
import glfw
import vulkan as vk
from pathlib import Path


path = Path(__file__).parent


class MyApp(App):

    def __init__(self):
        super().__init__('Hello World Example', n_images=10, n_frames=9)

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

        # Create a descriptor set and add the uniform buffer to it
        self.descriptor_set = DescriptorSet(self, n_images=self.swapchain.n_images)
        self.descriptor_set.add(self.uniforms, stages=vk.VK_SHADER_STAGE_VERTEX_BIT)
        self.descriptor_set.create()

        self.pass1 = Pass(self, clear_color=(0, 0, 0, 0))
        self.mesh1 = Drawable(
            self, triangle,
            vertex_shader=path / 'triangle.vert',
            fragment_shader=path / 'triangle.frag',
            render_pass=self.pass1,
            vertex_attributes=['vec2', 'vec3'],
            descriptor_sets=[self.descriptor_set]
        )
        
        self.last_time = glfw.get_time()
        self.last_count = 0
        self.fps_interval = 0.2

    def draw(self, image_i):
        with self.pass1.start(image_i, self.swapchain.images) as pass_ctx:
            self.mesh1.draw(pass_ctx)
        
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
