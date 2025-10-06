import time
import math
import numpy as np
from einops import rearrange, repeat
from omegaconf import OmegaConf
import torch
from torchvision.transforms import ToTensor

from .util import instantiate_from_config
from .data.dataset_camera_utils import get_pose_map

def get_unique_embedder_keys_from_conditioner(conditioner):
    return list(set([x.input_key for x in conditioner.embedders]))

def get_batch(keys, value_dict, N, T, device):
    batch = {}
    batch_uc = {}

    for key in keys:
        if key == "fps_id":
            batch[key] = (
                torch.tensor([value_dict["fps_id"]])
                .to(device)
                .repeat(int(math.prod(N)))
            )
        elif key == "motion_bucket_id":
            batch[key] = (
                torch.tensor([value_dict["motion_bucket_id"]])
                .to(device)
                .repeat(int(math.prod(N)))
            )
        elif key == "cond_aug":
            batch[key] = repeat(
                torch.tensor([value_dict["cond_aug"]]).to(device),
                "1 -> b",
                b=math.prod(N),
            )
        elif key == "cond_frames" or key == "cond_frames_without_noise":
            batch[key] = repeat(value_dict[key], "1 ... -> b ...", b=N[0])
        elif key == "polars_rad" or key == "azimuths_rad":
            batch[key] = torch.tensor(value_dict[key]).to(device).repeat(N[0])
        else:
            batch[key] = value_dict[key]

    if T is not None:
        batch["num_video_frames"] = T

    for key in batch.keys():
        if key not in batch_uc and isinstance(batch[key], torch.Tensor):
            batch_uc[key] = torch.clone(batch[key])
    return batch, batch_uc

def first_stage_sampling(model, value_dict, input_noise, num_frames, num_steps, device,
                         desired_angles, input_pos=0,
                         pose=np.array([0,0,0]),
                         max_deg_pos=15, max_deg_dir=8):

    z_img = model.encode_first_stage(value_dict["cond_frames"])
    z_nor = model.encode_first_stage(value_dict["cond_normals"])
    z = torch.cat([z_img, z_nor], dim=1)
    empty_image = model.encode_first_stage(torch.zeros_like(value_dict["cond_frames"])).repeat(1,2,1,1,)

    mask_loss = torch.ones(num_frames, device=device, dtype=torch.float)
    mask_loss[input_pos] = 0
    value_dict['mask_loss'] = mask_loss

    angles_y_padded = torch.cat([desired_angles, torch.zeros_like(desired_angles[:, [0]])], dim=1)
    condition_images = []
    rotations = []

    translation = torch.tensor([0., 0., 2.7], device=device, dtype=torch.float)
    for step, angle in enumerate(angles_y_padded):
        pose_maps, rotation = get_pose_map(torch.tensor(angle, dtype=torch.float), translation,
                                           max_deg_pos=max_deg_pos,
                                           max_deg_dir=max_deg_dir)
        mask_image = mask_loss[step, None, None, None].repeat(pose_maps.size(0), pose_maps.size(1), 1)
        condition_images.append(torch.cat([pose_maps.to(device), mask_image], dim=-1))
        rotations.append(rotation)

    condition_images = torch.stack(condition_images).permute(0, 3, 1, 2).contiguous()[None, ...]
    value_dict['cond_images'] = condition_images


    ## ------------------------------
    #          Start sampling
    ## ------------------------------
    with torch.no_grad():
        with torch.autocast(device):
            batch, batch_uc = get_batch(
                get_unique_embedder_keys_from_conditioner(model.conditioner),
                value_dict,
                [1, num_frames],
                T=num_frames,
                device=device,
            )
            batch_uc['txt'] = ['']
            c, uc = model.conditioner.get_unconditional_conditioning(
                batch,
                batch_uc=batch_uc,
                force_uc_zero_embeddings=[
                    # "cond_images"
                ],
            )
            uc['cond_images'][value_dict['mask_loss'].long() == 0] = 0
            for k in ["crossattn", "concat", ]:
                if k in c.keys():
                    uc[k] = repeat(uc[k], "b ... -> b t ...", t=num_frames)
                    uc[k] = rearrange(uc[k], "b t ... -> (b t) ...", t=num_frames)
                    c[k] = repeat(c[k], "b ... -> b t ...", t=num_frames)
                    c[k] = rearrange(c[k], "b t ... -> (b t) ...", t=num_frames)

            randn = input_noise.repeat(num_frames, 1, 1, 1)
            randn = torch.where(mask_loss[:, None, None, None] == 0., z, randn)

            additional_model_inputs = {}
            additional_model_inputs["image_only_indicator"] = torch.zeros(
                2, num_frames
            ).to(device)
            additional_model_inputs["num_video_frames"] = batch["num_video_frames"]
            additional_model_inputs["empty_image"] = empty_image

            def denoiser(input, sigma, c):
                return model.denoiser(
                    model.model, input, sigma, c, **additional_model_inputs
                )


            start_time = time.time()
            samples_z = model.sampler(denoiser, randn, cond=c, uc=uc,
                                      num_steps=num_steps,
                                      empty_image=empty_image, verbose=True)
            endtime = time.time()
            print(
                f'Processing time for retrieving shape normals: {endtime - start_time:0.4f} s')

    rendering_dict = {
        'condition_images': condition_images,
        'z': z,
    }
    return samples_z.detach(), rendering_dict

def sample_normals(model, input_image, id_emb, max_deg_pos=15, max_deg_dir=8):

    device = model.device
    z = model.encode_first_stage(input_image)
    translation = torch.tensor([0,0, 2.7], dtype=torch.float, device=device)
    pose_maps, _ = get_pose_map(torch.tensor([0., 0., 0.], dtype=torch.float, device=device), translation,
                                       max_deg_pos=max_deg_pos,
                                       max_deg_dir=max_deg_dir, device=device)

    normals_generation_sampler_config = OmegaConf.create({
        'target': 'spmr.modules.diffusionmodules.sampling.EulerEDMNormalsMaskedSamplerSpMR',
        'params':{
            'num_steps': 50,
            'discretization_config':{
                'target': 'spmr.modules.diffusionmodules.discretizer.LegacyDDPMDiscretization'
            },
            'guider_config':{
                    'target': 'spmr.modules.diffusionmodules.guiders.NoGuider',
                    'params':{
                    'scale': 1.
                },
        },
    }})
    normals_generation_sampler = instantiate_from_config(normals_generation_sampler_config)

    def denoiser(input, sigma, c):
        return model.denoiser(
            model.model, input, sigma, c, num_video_frames=1
        )


    mask_loss = torch.ones(1, device=device, dtype=torch.float)
    mask_image = torch.cat([pose_maps, mask_loss.expand(pose_maps.size(0), pose_maps.size(1), 1)], dim=-1)
    num_frames = 1

    value_dict = {}
    value_dict['txt'] = [id_emb]
    value_dict['mask_loss'] = mask_loss
    value_dict['cond_images'] = mask_image[None,None].permute(0,1,4,2,3)

    batch, batch_uc = get_batch(
        get_unique_embedder_keys_from_conditioner(model.conditioner),
        value_dict,
        [1, num_frames],
        T=num_frames,
        device=device,
    )
    batch_uc['txt'] = ['']
    c, uc = model.conditioner.get_unconditional_conditioning(
        batch,
        batch_uc=batch_uc,
        force_uc_zero_embeddings=[
        ],
    )

    randn = torch.randn((1, 8, z.size(-2), z.size(-1)), device=device)
    visibility_mask = torch.ones((1,8,z.size(-2),z.size(-1)), device=device, dtype=torch.float)
    visibility_mask[0,4:] = 0

    corrector_kwargs = {'gradient_weight': 100_000,
                        'gt_x0': z.repeat(1,2,1,1),
                        'visibility_mask': visibility_mask
                        }
    t1 = time.time()
    samples_latentcode = normals_generation_sampler(denoiser, randn.clone(), cond=c, uc=uc,
                                                    diffusion_model=model, verbose=False,
                                                    corrector_kwargs=corrector_kwargs,
                                                    )

    t2 = time.time()
    samples_x = model.decode_first_stage(samples_latentcode[:,:4])
    samples = torch.clamp(samples_x, min=-1.0, max=1.0)
    samples_x = model.decode_first_stage(samples_latentcode[:,4:])
    samples_normals = torch.clamp(samples_x, min=-1.0, max=1.0)
    t3 = time.time()
    print(f'Sampling processing time: {t2-t1:.2f} seconds, Decoding time {t3-t2:.2f} seconds, Total time {t3-t1:.2f} seconds')
    return samples, samples_normals


def sampling_pipeline(input_image, id_emb, model, desired_angles, scale=2.5, num_steps=50, num_frames=8, device='cuda', seed=42):

    print(f'{scale=} {num_frames=} {seed=}, {desired_angles=}')
    image = ToTensor()(input_image)
    image = image * 2.0 - 1.0


    image = image.unsqueeze(0).to(device)
    H, W = image.shape[2:]
    assert image.shape[1] == 3
    F = 8
    C = 8
    shape = (1, C, H // F, W // F)

    torch.manual_seed(seed)
    input_noise = torch.randn(shape, device=device)
    
    _, input_image_normals = sample_normals(model, image, id_emb)
    
    value_dict = {}
    value_dict['txt'] = [id_emb]
    value_dict["cond_frames_without_noise"] = image
    value_dict["cond_frames"] = image
    value_dict["cond_normals"] = input_image_normals

    model.sampler.guider.scale = scale
    start = time.time()
    samples_first_stage, rendering_dict = first_stage_sampling(model, value_dict, input_noise, num_frames,
                                                               num_steps, device, desired_angles=desired_angles)
    model.en_and_decode_n_samples_a_time = 4

    samples_x = model.decode_first_stage(samples_first_stage[:, :4])
    samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)

    samples_x_normals = model.decode_first_stage(samples_first_stage[:, 4:])
    samples_normals = torch.clamp((samples_x_normals + 1.0) / 2.0, min=0.0, max=1.0)
    end = time.time()
    print('Time taken to sample and decode : {} seconds'.format(end - start))
    
    return samples, samples_normals
 