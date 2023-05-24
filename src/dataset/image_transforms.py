import albumentations as A

def get_transforms(train=True,rain=None,sunny=None,snow=None,foggy=None):
    if train == True:
        Aug = [
                A.OneOf(
                    [
                        A.HueSaturationValue(
                            hue_shift_limit=0.1,
                            sat_shift_limit=0.1,
                            val_shift_limit=0.1,
                            p=0.7,
                        ),
                        A.RandomBrightnessContrast(
                            brightness_limit=0.15,
                            contrast_limit=0.15,
                            p=0.9
                        ),
                    ],
                    p=0.7,
                ),
                A.Cutout(
                    num_holes=8, max_h_size=24, max_w_size=24, fill_value=0, p=0.2
                ),
                A.MotionBlur(blur_limit=(3, 5), p=0),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Normalize(p=1.0),
            ]
        W = []
        if rain==True:
            # Rain effects
            W1 = A.RandomRain(
                slant_lower=-10,
                slant_upper=10,
                drop_length=15,
                drop_width=1,
                drop_color=(200, 200, 200),
                blur_value=12,
                brightness_coefficient=0.52,
                rain_type="heavy",
                p=0.25)
            W.append(W1)

            # Sunny effects.
        if sunny==True:
            W2 = A.RandomSunFlare(
                flare_roi=(0, 0, 1, 1),
                angle_lower=0,
                angle_upper=1,
                num_flare_circles_lower=4,
                num_flare_circles_upper=8,
                src_radius=100,
                src_color=(255, 255, 255),
                p=0.25)
            W.append(W2)
            # Snow effects.
        if snow==True:
            W3 = A.RandomSnow(
                snow_point_lower=0.1,
                snow_point_upper=0.3,
                brightness_coeff=2.5,
                p=0.25)
            W.append(W3)
            # Fog effects.
        if foggy==True:
            W4 = A.RandomFog(
                fog_coef_lower=0.3,
                fog_coef_upper=0.4,
                alpha_coef=0.1,
                p=0.01)
            W.append(W4)
        return A.Compose(Aug+W)
    else:
        return A.Compose([
            A.Normalize(p=1.0)])
