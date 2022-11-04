class Config:

    class Camera:
        width = int(640)
        height = int(480)
        fx = float(618.0714111328125)
        fy = float(617.783447265625)
        cx = float(305.902252197265625)
        cy = float(246.352935791015625)

    class Depth:
        lower_bound = float(0.3)
        upper_bound = float(1.0)

    class Module:
        gpu_id = int(0)
        period = float(0.1)
