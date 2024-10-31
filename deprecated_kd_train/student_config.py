class PipelineConfig:
    def __init__(self):
        self.seed = 59
        self.learning_rate = 1e-5
        self.epochs = 30
        self.train_batch_size = 32
        self.test_batch_size = 64
        self.hidden_dim = 2048
        self.projection_dim = 2048
        self.weight_decay = 1e-5
        self.patience = 3
        self.text_max_len = 50
        self.fusion_strategy = "concat+smalllen"
        self.text_encoder_id = "vinai/bartpho-word"
        self.img_encoder_id = "timm/resnet18.a1_in1k"
        self.paraphraser_id = "chieunq/vietnamese-sentence-paraphase"
        self.is_text_augment = True
        self.n_text_paras = 3
        self.text_para_thresh = 0.5
        self.n_text_para_pool = 30
        self.is_img_augment = False
        self.n_img_augments = 0
        self.img_augment_thresh = 0
        self.use_dynamic_thresh = False
        self.dataset_name = 'vivqa'
        self.data_dir = "/home/VLAI/datasets"
        self.use_amp = True
        self.pretrained_teacher_path = '/home/VLAI/khoina/VQA_Research/runs/train/weights_20241019-124853/Teacher_seed59_vivqa_currFalse&0.6_istextaugTrue_ntextpara1_random0.6_nparapool30_isimgaugTrue_nimgaug1_random0.2_teacher_best.pt'

pipeline_config = PipelineConfig()