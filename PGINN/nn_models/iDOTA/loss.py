import tensorflow as tf
from keras import losses
from keras.saving import register_keras_serializable


@register_keras_serializable(package="loss")
class TopPMSE(tf.keras.metrics.Metric):
    def __init__(self, p=0.1, name="top_p_mse", **kwargs):
        super().__init__(name=f'{name}_{p}', **kwargs)
        self.p = p
        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)

        if self.p <= 0:
            mask = tf.ones_like(y_true, dtype=tf.float32)
        else:
            n = tf.cast(tf.size(y_true), tf.float32)
            k = tf.cast(tf.math.ceil((1.0 - self.p) * n), tf.int32)
            threshold = tf.sort(y_true)[k]
            mask = tf.cast(y_true >= threshold, tf.float32)

        se = mask * tf.square(y_true - y_pred)
        self.total.assign_add(tf.reduce_sum(se))
        self.count.assign_add(tf.reduce_sum(mask))

    def result(self):
        return self.total / (self.count + 1e-8)

    def reset_state(self):
        self.total.assign(0.0)
        self.count.assign(0.0)

    def get_config(self):
        config = super().get_config()
        config.update({
            "p": self.p,
        })
        return config


@register_keras_serializable(package="loss")
class WeightedMSE(losses.Loss):
    def __init__(self, reduction="sum_over_batch_size", name="weighted_mse"):
        super().__init__(reduction=reduction, name=name)
        self.mse = losses.MeanSquaredError(reduction=reduction)

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        weights = y_true

        return self.mse(y_true, y_pred, sample_weight=weights)


@register_keras_serializable(package="loss")
class TopPMSELoss(losses.Loss):
    def __init__(self, p=0.1, reduction="sum_over_batch_size", name="top_p_mse"):
        super().__init__(reduction=reduction, name=name)
        self.p = p

    def call(self, y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)

        if self.p <= 0:
            mask = tf.ones_like(y_true, dtype=tf.float32)
        else:
            n = tf.cast(tf.size(y_true), tf.float32)
            k = tf.cast(tf.math.ceil((1.0 - self.p) * n), tf.int32)
            threshold = tf.sort(y_true)[k]
            mask = tf.cast(y_true >= threshold, tf.float32)

        se = mask * tf.square(y_true - y_pred)
        return tf.reduce_sum(se) / (tf.reduce_sum(mask) + 1e-8)

    def get_config(self):
        config = super().get_config()
        config.update({"p": self.p})
        return config

@register_keras_serializable(package="loss")
class SharpLoss(losses.Loss):
    def __init__(self, gamma=0.0, reduction="sum_over_batch_size", name="sharp_loss"):
        super().__init__(reduction=reduction, name=name)
        self.gamma = gamma
        self.mse = losses.MeanSquaredError(reduction="sum_over_batch_size", name='mse')

    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.convert_to_tensor(y_true, dtype=y_pred.dtype)
        y_true = tf.reshape(y_true, tf.shape(y_pred))

        # Return standard MSE if gamma is 0
        if self.gamma < 1e-8:
            return self.mse(y_true, y_pred)

        # Sharp loss modification as defined by Bai et al. https://pmc.ncbi.nlm.nih.gov/articles/PMC8501531/
        sigmoid = tf.nn.sigmoid(y_true * self.gamma)
        sq_sigmoid = sigmoid * tf.square(y_true - y_pred)
        return tf.reduce_mean(sq_sigmoid, axis=-1)


def gradient_3d(x: tf.Tensor) -> tf.Tensor:
    """
    Vectorized central differences along depth, height, width for channels-last 3D tensors.
    Input:  [B, D, H, W, C]
    Output: [3, B, D, H, W, C]  (axis 0 = dz, dy, dx)
    """
    # Pad symmetrically along each spatial axis
    x_pad = tf.pad(x, [[0,0],[1,1],[1,1],[1,1],[0,0]], mode="SYMMETRIC")

    # Depth differences
    dz = (x_pad[:, 2:, 1:-1, 1:-1, :] - x_pad[:, :-2, 1:-1, 1:-1, :]) / 2.0
    # Height differences
    dy = (x_pad[:, 1:-1, 2:, 1:-1, :] - x_pad[:, 1:-1, :-2, 1:-1, :]) / 2.0
    # Width differences
    dx = (x_pad[:, 1:-1, 1:-1, 2:, :] - x_pad[:, 1:-1, 1:-1, :-2, :]) / 2.0

    return tf.stack([dz, dy, dx], axis=0)

@register_keras_serializable(package="loss")
class DualGradientL2Loss(losses.Loss):
    """
    Gradient-weighted loss with two gamma values for channels-last 3D data:
      gamma_edge > 0 : focuses on steep GT gradients
      gamma_flat = 0 : matches gradients everywhere
    Total = lambda_edge * L_edge + lambda_flat * L_flat
    """
    def __init__(self,
                 gamma_edge=100,
                 gamma_flat=0,
                 lambda_edge=0.05,
                 lambda_flat=0.02,
                 reduction="sum_over_batch_size",
                 name="dual_gradient_l2_loss"):
        super().__init__(reduction=reduction, name=name)
        self.gamma_edge = gamma_edge
        self.gamma_flat = gamma_flat
        self.lambda_edge = lambda_edge
        self.lambda_flat = lambda_flat

    @staticmethod
    def _sharp_term(gp, gg, gamma):
        # gp, gg: [3, B, D, H, W, C]
        diff2 = tf.reduce_sum(tf.square(gp - gg), axis=0)  # [B, D, H, W, C]
        weight = tf.pow(tf.sqrt(tf.reduce_sum(tf.square(gg), axis=0)) + 1e-6, gamma)
        # weight = tf.clip_by_value(weight, clip_value_min=0.0, clip_value_max=6.0)
        return diff2 * weight

    def call(self, y_true, y_pred):
        gp = gradient_3d(y_pred)
        gg = gradient_3d(y_true)

        l_edge = self._sharp_term(gp, gg, self.gamma_edge)
        l_flat = self._sharp_term(gp, gg, self.gamma_flat)

        loss = self.lambda_edge * l_edge + self.lambda_flat * l_flat
        return tf.reduce_mean(loss) if self.reduction == tf.losses.Reduction.SUM_OVER_BATCH_SIZE else tf.reduce_sum(loss)

@register_keras_serializable(package="loss")
class MomentLoss(losses.Loss):
    """
    Matches the first N raw moments (mean, variance, …) over the entire volume.
    loss = Σ_n | μ_n(pred) - μ_n(gt) |
    
    Works with channels-last 3D tensors: [B, D, H, W, C]
    """
    def __init__(self, moments=(1, 2), reduction="sum_over_batch_size", name="moment_loss"):
        super().__init__(reduction=reduction, name=name)
        self.moments = moments

    @staticmethod
    def _moment(x, n):
        """Compute the n-th raw moment over all voxels."""
        return tf.reduce_mean(tf.pow(x, n))

    def call(self, y_true, y_pred):
        losses = []
        for n in self.moments:
            m_pred = self._moment(y_pred, n)
            m_true = self._moment(y_true, n)
            losses.append(tf.abs(m_pred - m_true))
        return tf.add_n(losses) / len(losses)

@register_keras_serializable(package="loss")
class MomentPlusDualGradientLoss(losses.Loss):
    def __init__(self,
                 moments=(1, 2),
                 moment_weight=1.0,
                 gamma_edge=100,
                 gamma_flat=0,
                 lambda_edge=0.05,
                 lambda_flat=0.02,
                 reduction="sum_over_batch_size",
                 name="moment_plus_dualgrad_loss"):
        super().__init__(reduction=reduction, name=name)
        self.moment_loss_fn = MomentLoss(moments=moments, reduction=tf.keras.losses.Reduction.NONE)
        self.moment_weight = moment_weight
        self.gamma_edge = gamma_edge
        self.gamma_flat = gamma_flat
        self.lambda_edge = lambda_edge
        self.lambda_flat = lambda_flat

    @staticmethod
    def _sharp_term(gp, gg, gamma):
        diff2 = tf.reduce_sum(tf.square(gp - gg), axis=0)  # [B, D, H, W, C]
        weight = tf.pow(tf.sqrt(tf.reduce_sum(tf.square(gg), axis=0)) + 1e-6, gamma)
        weight = tf.clip_by_value(weight, 0.0, 6.0)
        return diff2 * weight

    def call(self, y_true, y_pred):
        # Moment loss (scalar)
        moment_loss_val = self.moment_loss_fn(y_true, y_pred)

        # Gradient loss
        gp = gradient_3d(y_pred)
        gg = gradient_3d(y_true)
        l_edge = self._sharp_term(gp, gg, self.gamma_edge)
        l_flat = self._sharp_term(gp, gg, self.gamma_flat)
        grad_loss_val = self.lambda_edge * l_edge + self.lambda_flat * l_flat
        grad_loss_val = tf.reduce_mean(grad_loss_val) if self.reduction == tf.losses.Reduction.SUM_OVER_BATCH_SIZE else tf.reduce_sum(grad_loss_val)

        # Combine
        return self.moment_weight * moment_loss_val + grad_loss_val


@register_keras_serializable(package="loss")
class CombinedLoss(losses.Loss):
    """
    Combines:
      - MomentLoss (global raw moments)
      - DualGradientL2Loss (edge + flat gradient matching)
      - Standard MSE
    """
    def __init__(self,
                 moments=(1, 2),
                 moment_weight=1.0,
                 gamma_edge=100,
                 gamma_flat=0,
                 lambda_edge=0.05,
                 lambda_flat=0.02,
                 mse_weight=1.0,
                 reduction="sum_over_batch_size",
                 name="combined_loss"):
        super().__init__(reduction=reduction, name=name)
        # Moment loss with no internal reduction so we can combine before final reduce
        self.moment_loss_fn = MomentLoss(moments=moments, reduction=tf.keras.losses.Reduction.NONE)
        self.moment_weight = moment_weight
        self.gamma_edge = gamma_edge
        self.gamma_flat = gamma_flat
        self.lambda_edge = lambda_edge
        self.lambda_flat = lambda_flat
        self.mse_weight = mse_weight
        # MSE also with no internal reduction
        self.mse_fn = losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)

    @staticmethod
    def _sharp_term(gp, gg, gamma):
        diff2 = tf.reduce_sum(tf.square(gp - gg), axis=0)  # [B, D, H, W, C]
        weight = tf.pow(tf.sqrt(tf.reduce_sum(tf.square(gg), axis=0)) + 1e-6, gamma)
        weight = tf.clip_by_value(weight, 0.0, 6.0)
        return diff2 * weight

    def call(self, y_true, y_pred):
        # --- Moment loss ---
        moment_loss_val = self.moment_loss_fn(y_true, y_pred)  # scalar

        # --- Gradient loss ---
        gp = gradient_3d(y_pred)
        gg = gradient_3d(y_true)
        l_edge = self._sharp_term(gp, gg, self.gamma_edge)
        l_flat = self._sharp_term(gp, gg, self.gamma_flat)
        grad_loss_val = self.lambda_edge * l_edge + self.lambda_flat * l_flat
        grad_loss_val = tf.reduce_mean(grad_loss_val) if self.reduction == tf.losses.Reduction.SUM_OVER_BATCH_SIZE else tf.reduce_sum(grad_loss_val)

        # --- MSE loss ---
        mse_val = self.mse_fn(y_true, y_pred)  # per-example scalar
        mse_val = tf.reduce_mean(mse_val) if self.reduction == tf.losses.Reduction.SUM_OVER_BATCH_SIZE else tf.reduce_sum(mse_val)

        # --- Combine ---
        total_loss = (
            self.moment_weight * moment_loss_val +
            grad_loss_val +
            self.mse_weight * mse_val
        )
        return total_loss
