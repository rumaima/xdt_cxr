from torch.nn import Conv2d, Dropout
import math
import os.path as osp
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import load_checkpoint
from dassl.data.data_manager import DataManager
# from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer  # for CLIP
# from clip.clip import tokenize                                   # for CLIP
# _tokenizer = _Tokenizer()                                        # for CLIP
from utils.model_utils import *
from utils.utils import *
from functools import reduce
from operator import mul
from utils.data_utils import ds_specific_templates
from MedCLIP.medclip import MedCLIPModel, MedCLIPVisionModelViT, MedCLIPTextModel
from MedCLIP.medclip.dataset import MedCLIPProcessor


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url, root='all_weights')

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class LaFTerUFT(nn.Module):

    def __init__(self, model, classes, templates, ds_templates=None, device='cuda', log=None, dataset_name=None, txt_cls=None, cfg=None, processor=None):
        super(LaFTerUFT, self).__init__()
        # self.adapter_pl = None
        self.device = device
        self.cfg = cfg
        self.processor = processor
        self.dataset_templates = ds_templates
        self.classes = classes
        self.dataset_name = dataset_name
        self.classes = classes
        self.txt_cls = txt_cls
        self.log = log
        patch_size = (16,16)  #(4,4) for vitb16
        self.model = model.to(device)
        self.templates = templates
        self.backbone_out_size = 512 # todo for ViT-L use 768 - for ViT-B use 512
        self.hidden_size = 96 # todo for ViT-L use 1024 - for ViT-B use 768  -- for medclip swin 96
        self.num_tokens = 113 # todo all experiments are run with num_tokens = 50 b/32 = 257 l/14  -- for medclip swin 113 (224,224), 199 (396,396)
        self.prompt_proj = nn.Identity()
        self.hidden_size_text = 512 #512 remains the same
        self.num_tokens_text = 77
        prompt_dim = self.hidden_size
        val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + prompt_dim))  # noqa
        self.prompt_dropout = Dropout(0.0)
        self.adapter = nn.Sequential(nn.Linear(int(self.backbone_out_size), len(classes), bias=False)).to(device)
        self.prompt_embeddings = nn.Parameter(torch.zeros(
            1, self.num_tokens, self.hidden_size), requires_grad=True)
        nn.init.uniform_(self.prompt_embeddings.data, -val, val)
        self.txt_features_for_text_cls, self.labels_for_text_cls = self.txt_features_for_text_cls()
        self.text_features = self.txt_features()
        self.classifier = nn.Sequential(nn.Linear(16, len(classes), bias=False)).to(device)
        # self.dim_reduction = nn.Sequential(nn.Linear(512, 16, bias=False)).to(device)
        self.adapter_pl = nn.Sequential(nn.Linear(int(self.backbone_out_size), len(classes), bias=False)).to(device)
        
        
    def train_txt_clas(self, criteria):
        noise_std = 0.1
        noise = torch.randn(self.txt_features_for_text_cls.shape) * noise_std
        txt_feas = self.txt_features_for_text_cls
        txt_label = self.labels_for_text_cls
        feas = (self.adapter(txt_feas.to(torch.float32) + noise.cuda()))
        loss = criteria(feas, txt_label)
        return loss
    
    def test_txt_clas(self, image):
        noise_std = 0.1
        noise = torch.randn(self.txt_features_for_text_cls.shape) * noise_std
        txt_feas = self.image_features(image)
        txt_label = self.labels_for_text_cls
        feas = (self.adapter(txt_feas.to(torch.float32)))
        return feas

    def txt_features_for_text_cls(self):

        if self.txt_cls== 'cls_only':
            gpt3_prompts = None
            desc, labels_for_descriptions = gen_labels_with_classes(self.classes, descriptions=gpt3_prompts)

        elif self.txt_cls == 'templates_only':
            gpt3_prompts = self.templates
            desc, labels_for_descriptions = gen_labels_with_templates(self.classes, descriptions=gpt3_prompts)

        elif self.txt_cls == 'lafter':
            # generic prompts + templates

            if self.dataset_name not in lafter_datasets:
                raise ValueError('Invalid dataset name for LaFTer')

            path_to_file = f'./descriptions/generic/{self.dataset_name}.json'
            with open(path_to_file) as f:
                gpt3_prompts = json.load(f)

            desc, labels_for_descriptions = gen_labels_with_descrptions(self.classes, descriptions=gpt3_prompts)

            templates, labels_for_templates = gen_labels_with_templates(self.classes, descriptions=self.dataset_templates)

            desc += templates
            labels_for_descriptions += labels_for_templates

        elif self.txt_cls == 'zero_shot':
            pass

        else:
            raise ValueError('Invalid txt_cls argument')


        if self.txt_cls in ['cls_only', 'templates_only', 'lafter']:

            Path(f'embeddings').mkdir(parents=True, exist_ok=True)

            if os.path.isfile(f'embeddings/{self.txt_cls}_{self.dataset_name}_embeddings.pt'):
                zeroshot_weights = torch.load(f'embeddings/{self.txt_cls}_{self.dataset_name}_embeddings.pt')
                print('******** Loaded Already Saved Embeddings *********')
                labels_for_descriptions = torch.tensor(labels_for_descriptions).cuda()

            else:
                print('******** No Embeddings Found --- Saving New Embeddings *********')

                labels_for_descriptions = torch.tensor(labels_for_descriptions).cuda()

                zeroshot_weights = []
                with torch.no_grad():
                    for classname in tqdm(desc):
                        prompt_texts = [template.format(classname) for template in self.templates]  # format with class
                        # texts = tokenize(prompt_texts).cuda()  # tokenize # (50, 77) --> 50 templates/texts from GPT for CLIP
                        texts = self.processor(text=prompt_texts ,return_tensors="pt", padding=True).input_ids.cuda() # tokenize # (50, 77) --> 50 templates/texts from GPT for MedCLIP
                        class_embeddings = self.model.encode_text(texts)  # embed with text encoder # (50, 512) --> embeddings for all 50 texts
                        class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)  # L2 norm of the embeddings (dim 2)
                        zeroshot_weights.append(class_embeddings)
                    zeroshot_weights = torch.stack(zeroshot_weights).cuda()  # (512, 10) --> 512 embeddings for 10 classes'
                    torch.save(zeroshot_weights, f'embeddings/{self.txt_cls}_{self.dataset_name}_embeddings.pt')

            return zeroshot_weights.squeeze(), labels_for_descriptions

        else:
            return None, None

    def txt_features(self):
        with torch.no_grad():
            zeroshot_weights = []
            for classname in tqdm(self.classes):
                prompt_texts = [template.format(classname) for template in self.templates]  # format with class
                # texts = tokenize(prompt_texts).cuda()  # tokenize for CLIP
                texts = self.processor(text=prompt_texts ,return_tensors="pt", padding=True).input_ids.cuda() # tokenize for MedCLIP
                class_embeddings = self.model.encode_text(texts)  # embed with text encoder
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)
            zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
        return zeroshot_weights


    def image_features(self, images):
        with torch.no_grad():
            image_features = self.model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            return image_features

    def textual_features(self, classes):
        with torch.no_grad():
            zeroshot_weights = []
            for classname in tqdm(classes):
                prompt_texts = [template.format(classname) for template in self.templates]  # format with class
                # texts = tokenize(prompt_texts).cuda()  # tokenize for CLIP
                texts = self.processor(text=prompt_texts ,return_tensors="pt", padding=True).input_ids.cuda() # tokenize for MedCLIP
                class_embeddings = self.model.encode_text(texts)  # embed with text encoder
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)
            zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
            return zeroshot_weights

    def eval_clip(self, x):
        with torch.no_grad():
            # img_features_2 = self.incorporate_prompt(x)
            # img_features_2 = self.embeddings_after_prompts(img_features_2) for clip
            img_features_2, dimension_size_2 = self.incorporate_prompt(x)
            img_features_2 = self.embeddings_after_prompts(img_features_2, dimension_size_2)
            img_features_adapter = self.adapter(img_features_2)
        return img_features_adapter

    def eval_clip_with_embeddings(self, x):
        with torch.no_grad():
            # img_features_2 = self.incorporate_prompt(x)
            # img_features_2 = self.embeddings_after_prompts(img_features_2) for clip
            img_features_2, dimension_size_2 = self.incorporate_prompt(x)
            img_features_2 = self.embeddings_after_prompts(img_features_2, dimension_size_2)
            img_features_adapter = self.adapter(img_features_2)
        return img_features_2, img_features_adapter

    def forward(self, x):
        # only used for 0-shot-eval
        with torch.no_grad():
            img_features = self.image_features(x)
            pseudo_label = img_features @ self.text_features
        return pseudo_label

    def forward_normal_evaluate_no_prompts(self,x, model_t):
        # Load the adapter weights
        img_features_1 = self.image_features(x)
        transformer_embed = model_t(img_features_1)
        pseudo_label = self.classifier(transformer_embed.float()).detach()
        pseudo_label_softmax = F.softmax(pseudo_label, dim=-1)
        return pseudo_label_softmax

    def forward_aug_with_prompts(self, x2):
        '''
        :param x1: the clean image (without transforms, for pseudo labels, for teacher)
        :param x2: the transformed image (for student)
        :return: features adapter (cls head), pseudo-labels
        '''
        # img_features_2 = self.incorporate_prompt(x2)
        # img_features_2 = self.embeddings_after_prompts(img_features_2) for clip
        img_features_2, dimension_size_2 = self.incorporate_prompt(x2)
        img_features_2 = self.embeddings_after_prompts(img_features_2, dimension_size_2)
        img_features_adapter = self.adapter(img_features_2)
        return img_features_adapter

    def txt_cls_init(self):
        import copy
        self.adapter_pl = copy.deepcopy(self.adapter)

    def forward_normal_for_pl(self, x1):
        '''
        :param x1: the clean image (without transforms, for pseudo labels, for teacher)
        :param x2: the transformed image (for student)
        :return: features adapter (cls head), pseudo-labels
        '''
        with torch.no_grad():
            img_features_1 = self.image_features(x1)
            pseudo_label = self.adapter_pl(img_features_1.float()).detach()
        return pseudo_label

    def forward_normal_no_prompts(self, x):
        '''
        :param x1: the clean image (without transforms, for pseudo labels, for teacher)
        :param x2: the transformed image (for student)
        :return: features adapter (cls head), pseudo-labels
        '''
        with torch.no_grad():
            img_features = self.image_features(x)
        return img_features

    def forward_mlhc_mlp(self,x, model_t):
        with torch.no_grad():
            img_features_1 = self.image_features(x)
            pseudo_label_softmax = model_t(img_features_1)
            return pseudo_label_softmax

    def incorporate_prompt(self, x, teacher=False):
        B = x.shape[0]
        # x = self.patch_embeddings(x)  # (batch_size, 1 + n_patches, hidden_dim) ## for clip
        x, y = self.patch_embeddings(x) # for medclip

        x = torch.cat((
            x[:, :1, :],
            self.prompt_dropout(self.prompt_proj(self.prompt_embeddings).expand(B, -1, -1)),
            x[:, 1:, :]
        ), dim=1)
        return x, y # for medclip and only x for clip

    def patch_embeddings(self, x: torch.tensor): 
        # shape of x = torch.Size([50, 3, 224, 224]) for CLIP
        # self.model.model(x)
        embedding_output, input_dimensions = self.model.vision_model.model.embeddings(x) # for MEDCLIP  
        return embedding_output, input_dimensions
        # self.model.vision_model.model.embeddings(x)[0].shape  torch.Size([50,3136,96])
        # return self.model.visual.embeddings_patch(x) # for CLIP torch.Size([50, 50, 768])

    def embeddings_after_prompts(self, x: torch.tensor, input_dimensions): 
        
        bool_masked_pos = None
        head_mask = None
        output_attentions = None
        output_hidden_states = None
        return_dict = None

        output_attentions = output_attentions if output_attentions is not None else self.model.vision_model.model.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.model.vision_model.model.config.output_hidden_states
        )
        # return_dict = return_dict if return_dict is not None else self.model.vision_model.model.config.use_return_dict

        if x is None:
            raise ValueError("You have to specify pixel_values")

        project = True
        pooled_output = None

        head_mask = self.model.vision_model.model.get_head_mask(head_mask, len(self.model.vision_model.model.config.depths))

        encoder_outputs = self.model.vision_model.model.encoder(
            x,
            (57,57),                                      # (57,57) for 224 and (100,100) for 396
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = encoder_outputs[0]
        sequence_output = self.model.vision_model.model.layernorm(sequence_output)

        
        if self.model.vision_model.model.pooler is not None:
            pooled_output = self.model.vision_model.model.pooler(sequence_output.transpose(1, 2))
            pooled_output = torch.flatten(pooled_output, 1)

        if not return_dict:
            output = (sequence_output, pooled_output) + encoder_outputs[1:]
        output_dict = {'last_hidden_state': output[0], 'pooler_output': output[1]}
        img_embeds = output_dict['pooler_output']
        if project:
            img_embeds = self.model.vision_model.projection_head(img_embeds)
        return img_embeds
        # shape of x = torch.Size([50, 100, 768]) for CLIP
        # return self.model.visual.forward_after_patch_embeddings(x)  # for CLIP torch.Size([50, 512])

    def positional_embeddings_for_text(self, x: torch.tensor):
        return self.model.text_model.model.embeddings.position_embeddings(x)
        # return self.model.positional_embeddings(x) for CLIP 

    def embeddings_after_prompts_for_text(self, x: torch.tensor):
        return self.model.embeddings_after_prompting(x)


@TRAINER_REGISTRY.register()
class LaFTer(TrainerX):

    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        # ### Loading CLIP pretrained weights
        # print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        # clip_model = load_clip_to_cpu(cfg)
        # if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
        #     clip_model.float()
        # print("Building ZERO-SHOT-MODEL CLIP")
        # self.model = LaFTerUFT(model=clip_model, classes=classnames,
        #                                   templates=['a photo of a {}'], ds_templates = ds_specific_templates[cfg.DATASET.NAME], dataset_name= cfg.DATASET.NAME, txt_cls = cfg.txt_cls, cfg=cfg)
        
        ### Loading MedCLIP pretrained weights
        processor = MedCLIPProcessor()
        print(f"Loading MedCLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        medclip_model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
        medclip_model.from_pretrained()
        medclip_model.cpu()
        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            medclip_model.float()
        print("Building ZERO-SHOT-MODEL MEDCLIP")
        self.model = LaFTerUFT(model=medclip_model, classes=classnames,
                                          templates=['a photo of a {} chest x_ray'], ds_templates = ds_specific_templates[cfg.DATASET.NAME], 
                                          dataset_name= cfg.DATASET.NAME, txt_cls = cfg.txt_cls, cfg=cfg,
                                          processor=processor)
        self.register_model("adapt", self.model)
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def build_data_loader(self):
        """Create essential data-related attributes.

        A re-implementation of this method must create the
        same attributes (self.dm is optional).
        """
        dm = DataManager(self.cfg, custom_tfm_test=te_transform, custom_tfm_train=tr_transforms)

        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u  # optional, can be None
        self.val_loader = dm.val_loader  # optional, can be None
        self.test_loader = dm.test_loader

        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname  # dict {label: classname}

        self.dm = dm

    def parse_batch_train(self, batch):

        if isinstance(batch, list):
            input = batch["img"]
            input = torch.stack(input)  # two views from dataloader
            input = input.to(self.device)
        else:
            input = batch['img']
            input = input.to(self.device)

        label = batch["label"]
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)