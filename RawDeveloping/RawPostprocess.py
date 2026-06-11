import torch
import torch.nn as nn


class RawPostprocess(nn.Module):
    def __init__(
        self,
        gamma=1 / 2.2,
        alpha=5.5,
        beta=0.4,
        exposure=1.0,
        white=1.0,
        black=0.0,
        r=1.0,
        g=1.0,
        b=1.0,
        sat=1.0,
        method="logistic",
        learnable=False,
    ):
        super().__init__()

        self.method = method

        def make_param(x):
            x = torch.tensor(float(x))
            if learnable:
                return nn.Parameter(x)
            return x

        self.gamma = make_param(gamma)
        self.alpha = make_param(alpha)
        self.beta = make_param(beta)

        self.exposure = make_param(exposure)

        self.white = make_param(white)
        self.black = make_param(black)

        self.r = make_param(r)
        self.g = make_param(g)
        self.b = make_param(b)

        self.sat = make_param(sat)

    def apply_wb(self, x):
        rgb = torch.stack([self.r, self.g, self.b])
        rgb = rgb / rgb.mean()

        rgb = rgb.view(1, 3, 1, 1)

        return torch.clamp(x * rgb, 0.0, 1.0)

    def white_and_black_point(self, x):
        x = x - self.black

        slope = 1.0 / (self.white - self.black + 1e-8)

        x = x * slope

        return torch.clamp(x, 0.0, 1.0)

    def apply_exposure(self, x):
        return torch.clamp(x * self.exposure, 0.0, 1.0)

    def apply_saturation(self, x):
        mean = x.mean(dim=1, keepdim=True)

        x = (x - mean) * self.sat + mean

        return torch.clamp(x, 0.0, 1.0)

    def apply_gamma(self, x):
        return torch.clamp(x, 0.0, 1.0) ** self.gamma

    def apply_parametric_s_curve(self, x):
        x = torch.clamp(x, 0.0, 1.0)

        if self.method == "logistic":

            alpha = torch.clamp(self.alpha, min=1e-6)

            def sigmoid(v):
                return torch.sigmoid(alpha * (v - self.beta))

            gx = sigmoid(x)
            g0 = sigmoid(torch.tensor(0.0, device=x.device, dtype=x.dtype))
            g1 = sigmoid(torch.tensor(1.0, device=x.device, dtype=x.dtype))

            out = (gx - g0) / (g1 - g0 + 1e-8)

        elif self.method == "rational":

            eps = 1e-7

            alpha = torch.clamp(self.alpha, min=eps)

            x_safe = x.clamp(eps, 1.0 - eps)

            num = (
                (x_safe**alpha)
                * ((1.0 - self.beta) ** alpha)
            )

            den = (
                num
                + ((1.0 - x_safe) ** alpha)
                * (self.beta**alpha)
            )

            out = num / (den + eps)

        else:
            raise ValueError(
                f"Unknown method '{self.method}'"
            )

        return torch.clamp(out, 0.0, 1.0)

    def forward(self, x):

        squeeze = False

        if x.ndim == 3:
            x = x.unsqueeze(0)
            squeeze = True

        x = self.apply_exposure(x)
        x = self.apply_wb(x)
        x = self.apply_saturation(x)
        x = self.white_and_black_point(x)

        x = self.apply_gamma(x)
        x = self.apply_parametric_s_curve(x)

        if squeeze:
            x = x.squeeze(0)

        return x