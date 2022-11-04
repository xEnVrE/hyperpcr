class Config:

    class Camera:
        width = int(1280)
        height = int(720)
        fx = float(1229.4285612615463)
        fy = float(1229.4285612615463)
        cx = float(640)
        cy = float(360)

    class Depth:
        lower_bound = float(0.3)
        upper_bound = float(1.0)

    class Module:
        gpu_id = int(0)
        period = float(0.016)
