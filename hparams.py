from text import symbols

class HParams(object):
    def __init__(self, **hparams):
        self.hparamdict = hparams
        for k, v in hparams.items():
            setattr(self, k, v)

    def __repr__(self):
        return "HParams(" + repr([(k, v) for k, v in self.hparamdict.items()]) + ")"

    def __str__(self):
        return ','.join([(k + '=' + str(v)) for k, v in self.hparamdict.items()])

    def parse(self, params):
        for p in params.split(","):
            key, value = p.split("=", 1)
            key = key.strip()
            t = type(self.hparamdict[key])
            if t == bool:
                value = value.strip().lower()
                if value in ['true', '1']:
                    value = True
                elif value in ['false', '0']:
                    value = False
                else:
                    raise ValueError(value)
            else:
                value = t(value)
            self.hparamdict[key] = value
            setattr(self, key, value)
        return self

def create_hparams(hparams_string=None, verbose=False):
    """Create model hyperparameters. Parse nondefault from given string."""

    hparams = HParams(
        ################################
        # Experiment Parameters        #
        ################################
        epochs=500,
        iters_per_checkpoint=1000,
        seed=1234,
        dynamic_loss_scaling=True,
        fp16_run=False,
        distributed_run=False,
        dist_backend="nccl",
        dist_url="tcp://localhost:54321",
        cudnn_enabled=True,
        cudnn_benchmark=False,
        ignore_layers=['embedding.weight'],

        ################################
        # Data Parameters             #
        ################################
        load_mel_from_disk=False,
        training_files='filelists/hindi_train.txt',
        validation_files='filelists/hindi_test.txt',
        text_cleaners=['transliteration_cleaners'],

        ################################
        # Audio Parameters             #
        ################################
        max_wav_value=32768.0,
        sampling_rate=8000,
        filter_length=1024,
        hop_length=100,
        win_length=400,
        n_mel_channels=80,
        mel_fmin=0.0,
        mel_fmax=8000.0,

        ################################
        # Model Parameters             #
        ################################
        n_symbols=len(symbols),
        symbols_embedding_dim=512,

        # Encoder parameters
        encoder_kernel_size=5,
        encoder_n_convolutions=3,
        encoder_embedding_dim=512,

        # Decoder parameters
        n_frames_per_step=1,  # currently only 1 is supported
        decoder_rnn_dim=1024,
        prenet_dim=256,
        max_decoder_steps=1000,
        gate_threshold=0.5,
        p_attention_dropout=0.1,
        p_decoder_dropout=0.1,

        # Attention parameters
        attention_rnn_dim=1024,
        attention_dim=128,

        # Location Layer parameters
        attention_location_n_filters=32,
        attention_location_kernel_size=31,

        # Mel-post processing network parameters
        postnet_embedding_dim=512,
        postnet_kernel_size=5,
        postnet_n_convolutions=5,

        ################################
        # Optimization Hyperparameters #
        ################################
        use_saved_learning_rate=False,
        learning_rate=3e-4,
        weight_decay=1e-6,
        grad_clip_thresh=1.0,
        batch_size=32,
        mask_padding=True  # set model's padded outputs to padded values
    )

    if hparams_string:
        hparams.parse(hparams_string)

    return hparams
