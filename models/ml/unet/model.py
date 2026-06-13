import torch
from torch import nn                                                                     # Neural networks


def _make_norm(norm_type: str, num_channels: int, group_norm_groups: int) -> nn.Module:
    norm_key = (norm_type or "batch").lower()
    if norm_key == "batch":
        return nn.BatchNorm2d(num_channels)
    if norm_key == "instance":
        return nn.InstanceNorm2d(num_channels, affine=True)
    if norm_key == "group":
        groups = min(max(int(group_norm_groups), 1), num_channels)
        while num_channels % groups != 0 and groups > 1:
            groups -= 1
        return nn.GroupNorm(groups, num_channels)
    raise ValueError(f"Unsupported norm_type: {norm_type}")


class DoubleConv(nn.Module):                                                             # Vi definerer en blok med to convolution lag (normalt for en U-net)
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_type: str = "batch",
        group_norm_groups: int = 8,
    ):                                                                                    # modtager antal input og output kanaler
        super().__init__()                                                               # For at initialize parent classen (nn. Module)
        self.net = nn.Sequential(                                                        # Et sequential lag som så kører lagene i rækkefølge.
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),  # Det er første convolution, som er ern 3x3 kernel. Bruger padding = 1 så vi får samme samme spatial padding. 
            _make_norm(norm_type, out_channels, group_norm_groups),                      # Normaliserer features.
            nn.ReLU(inplace=True),                                                       # Bruge ReLU activation
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False), # Anden convolution...
            _make_norm(norm_type, out_channels, group_norm_groups),                      # Normalisering igen...
            nn.ReLU(inplace=True),                                                       # Relu....
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:                                  # Forward pass (hvordan data bliver sendt gennem blokken)
        return self.net(x)                                                               # Sender inputtet gennem DoubleConv blokken, og så bliver outputtet returneret så det kan bruges i det næste lav i UNet.forward()


class UNet(nn.Module):                                                                   # Her definerer vi hele Unet modellen !!!!
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        features: tuple[int, ...] = (64, 128, 256, 512),
        norm_type: str = "batch",
        group_norm_groups: int = 8,
    ): 
                                                                                         # in_channels er antal input kanaler - vi bruger RGB så her er det 3.
                                                                                         # out_channels er antal output kanaler - vi vil have binær segmentering så det er 1.
                                                                                         # features er antal feature maps (hvor stærkt et bestemt mønster (feature) findes i hvert punkt af inputtet efter en convolution)
        super().__init__()                                                               #initialise pytorches base model...
        self.down_blocks = nn.ModuleList()                                               # En liste over encoder blokke (downsamling). Input image 512×512×3, Efter første blok: 512×512×64, Efter pooling: 256×256×64, Efter næste blok: 256×256×128
        self.up_transpose = nn.ModuleList()                                              # Liste over transposed convolution laf (upsampling delen). Her gemmes lag der gør billedet større igen.
        self.up_blocks = nn.ModuleList()                                                 # Efter upsampling kører man igen: DoubleConv, for at lære features på den højere opløsning. 
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)                                # Halverer billed størelse. [[1,2],[3,4]] bliver til 4 (største og stærkeste feature)

        channels = in_channels                                                           # vi starter antal kanaler hvilket svarer til antal input kanaler.
        for feature in features:                                                         # Vi looper så over feature størrelser (64,128,256,512) (altså hvor mange slags ønstre modellen kan repræsentere på én gang). Efter en convolution kan vi gå fra 3 bilelder (RGB) til 64 f.eks. hvor hver repræsenterer en feature. 
            self.down_blocks.append(
                DoubleConv(channels, feature, norm_type=norm_type, group_norm_groups=group_norm_groups)
            )                                                                            # Vi tilføjer nu en encoder blok.
            channels = feature                                                           # Channel antal skal opdateres til næste lag da vi nu har fået flere.

        self.bottleneck = DoubleConv(
            features[-1],
            features[-1] * 2,
            norm_type=norm_type,
            group_norm_groups=group_norm_groups,
        )                                                                                # Vores bottleneck!! Midten af modellen woohoo, feature dimensionen fordobles.

        rev_features = list(reversed(features))                                          # reverser listen med feature størrelserne - nu skal vi jo op af U Net delen (decoder)
        up_channels = features[-1] * 2                                                   # Decoder starter med bottlenecks output channels
        for feature in rev_features:                                                     # Loop over størrelserne af decoder features
            self.up_transpose.append(nn.ConvTranspose2d(up_channels, feature, kernel_size=2, stride=2)) # ConvTranspose2d laver upsamplingen for os (altså fordobler bileld str. )
            self.up_blocks.append(
                DoubleConv(feature * 2, feature, norm_type=norm_type, group_norm_groups=group_norm_groups)
            )                                                                            # Samler to feature maps, så channels bliver summen af de to som er dobbelt så mange. 
            up_channels = feature                                                        # Nu skal vi lige opdatere antal channels til næste encoder lag.

        self.head = nn.Conv2d(features[0], out_channels, kernel_size=1)                  # Endelige convolution der mapper vores feature maps til vores output kanal som er en mask.

    def forward(self, x: torch.Tensor) -> torch.Tensor:                                  # kør forward pass gennem hele U net modellen !!
        skip_connections: list[torch.Tensor] = []                                        # Gemmer envoder outputs...

        for down in self.down_blocks:                                                    # For downsampling(encoder paths)
            x = down(x)                                                                  # vi kører DoubleConv blok
            skip_connections.append(x)                                                   # Gemmer vores Output til skip connection
            x = self.pool(x)                                                             # pooler (reducerer bileldets størrelse)

        x = self.bottleneck(x)                                                           # Kører vores blok med bottleneck
        skip_connections = skip_connections[::-1]                                        # Laver reverse på skip connections til det skal bruges til vores decoder

        for idx in range(len(self.up_transpose)):                                        # Upsampling!!!
            x = self.up_transpose[idx](x)                                                # upsampling på en feature map
            skip = skip_connections[idx]                                                 # Så henter vi den tilsvarende encoder feature map

            if x.shape[-2:] != skip.shape[-2:]:                                          # Lad os lige sørge for at spatial dimentions passer... ellers...
                x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False) # Vi resizer med Bilinear interpolation som jeg ikke er helt sikker på hvad er matematisk men har læst det er godt? Det er vist en lineær kombination af de nærmeste nabo-pixels i to dimensioner.

            x = torch.cat((skip, x), dim=1)                                              #sætter to tensorer sammen fir skip connecton og så decoder vi features...
            x = self.up_blocks[idx](x)                                                   # Kører DoubleConv på den feature map som vi nu har lavet som er kombineret

        return self.head(x)                                                              # Endelige convolution laver så output mask logits... 5 big booms...: BOOM, BOOM, BOOM, BOOM,... BOOOOOOOOOM!  
