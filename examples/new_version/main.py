from kilauea import App, Graph, Pass, Drawable, Compute, Swapchain, UniformBuffer, Executor()

uniforms = UniformBuffer()

n_images = 4
n_frames = 3

g = Graph()
swapchain = Swapchain()
p1 = Pass(wait_for=[swapchain], attachments=[uniforms])
p2 = Pass(wait_for=[p1], attachments=[uniforms])
p3 = Pass(wait_for=[swapchain], attachments=[uniforms])
p4 = Pass(wait_for=[p2, p3], attachments=[uniforms])
swapchain.configure(wait_for=[p4])

executor = Executor()


def record_commands():
    for i in range(n_images):
        for p in passes:
            p.draw()


def draw():
    frame = executor.get_next_frame()








