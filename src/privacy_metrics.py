from tensorflow_privacy.privacy.analysis.compute_dp_sgd_privacy_lib import compute_dp_sgd_privacy


def compute_epsilon(dataset_size, batch_size, noise_multiplier, epochs, delta=1e-5):

    eps, _ = compute_dp_sgd_privacy(
        n=dataset_size,
        batch_size=batch_size,
        noise_multiplier=noise_multiplier,
        epochs=epochs,
        delta=delta,
    )

    return eps