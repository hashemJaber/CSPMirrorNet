import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torchvision.transforms as transforms
import numpy as np


#NOTE TO SELF: USE SPARSE CONVOLUITONS IN EARLIER LAYERS ?????
class InputLayer(nn.Module):
   def __init__(self,target_height,target_width)->None:
      super(InputLayer,self).__init__()
      self.target_height = target_height
      self.target_width = target_width
      self.resize_transform = transforms.Resize((target_height, target_width))


   def forward(self, images:Tensor,transform_to_fixed:bool=False)-> Tensor:
       """
       input:
          transform_to_fixed: bool #True: to resize images not matching accepted fixed lenght, False: to use default images size
          images: Tensor of shape (B,C,W,H) #Batch size, Channel, Width, and Height
       output:
          images: Tensor of shape (B, C, W2,H2) #Batch size, Channel, Width (updated to accept fixed lenght), and Height(updated to accept fixed length)

       Note: transform_to_fixed default to False due to us adding the SPP layer later on, SPP will enbale to use most of the networks upto the head detection layer's neck with any size of image's width's and heights
       """
       if(transform_to_fixed):
          # Convert the tensor to PIL Image, apply the resize, and convert back to a tensor
          image = F.interpolate(images, size=(self.target_height, self.target_width), mode='bilinear', align_corners=False)
          return image

       return image



# Question: Should we sum after applying activations or before ?, then concatinate ?
# WOuld not change model performances, does not add any feaure map richness/complexity
class ConvBlockCSP(nn.Module):
   """
   ConvBlockCSP: Similar to CSPNet except in the layer we concatinate we add its values to the layer after it, and then concnatinate both of them.
   input:
      input_channel: int, number of channels to input
      out_put_channel: int, number of channels to output
      kernet_size: tuple (w,h), default: (1,1)
      stride:int, effect the (W-kenrnel_size[0]+2*padding)/stride
   output:
         original feature map, new feature map
   """
   def __init__(self, input_channel: int,out_put_channel:int,stride:int=2, kernel_size:tuple=(5,5), verbose:bool=False)->None:
       super(ConvBlockCSP,self).__init__()
       self.conv1= nn.Conv2d(input_channel,out_put_channel,kernel_size=kernel_size,stride=2,padding=1 )
       self.activation = nn.Mish()

       self.conv2= nn.Conv2d(out_put_channel,out_put_channel,kernel_size=kernel_size,stride=1 )
       self.activation2 = nn.Mish()

       self.bn = nn.BatchNorm2d(out_put_channel)

       self.bn2 = nn.BatchNorm2d(out_put_channel)
       self.verbose=verbose

       #TODO: Add batch normalization


   def forward(self, feauture_map) -> (Tensor, Tensor):
     x1 = self.bn(self.conv1(feauture_map))
     x1=self.activation(x1)

     x2 = self.bn2(self.conv2(x1))
     x2=self.activation(x2)
     #in case we change kernel size to !=(1,1), x1 needs to match 2 for summation, Fixed
     if x1.size(2) != x2.size(2) or x1.size(3) != x2.size(3):
            x1 = nn.functional.adaptive_avg_pool2d(x1, (x2.size(2), x2.size(3)))

     x3 = torch.cat([x1,x2+x1],1)

     if(self.verbose):
       print('x3 shape ',x3.shape)
     return (feauture_map, x3 )

#Take Part 1 and Part 2, then switch to the other parts, such as part 2, and part 1 for cspNet

# Idea: We generate a Parameter generating network to automaically solve for hyper parameters, we use it every 100 epochs as a loss, or maybe the lowest of 10 of all the 100 epochs
class BaseConvBlockCSP(nn.Module):
  """
  BaseConvBlockCSP: Will be fixed soon
   input:
         Feautrue Map: Tensor

   output:
         tuple: (Tensor,Tensor)
  """
  def __init__(self,input_channel: int,out_put_channel:int,num_blocks:int=3,verbose:bool=False,use_single_layer:bool=True)->None:

    super(BaseConvBlockCSP,self).__init__()
    self.verbose=verbose

    self.ConvBlockCSP= ConvBlockCSP(input_channel,out_put_channel) # In=3, out=8
    ###Left for debugging purposes
    self.ConvBlockCSP2= ConvBlockCSP(out_put_channel*2,out_put_channel*4) #in= 8 out_put_channel*2, out=32
    self.ConvBlockCSP3= ConvBlockCSP(out_put_channel*4,out_put_channel*6)#in 32, out= 48
    self.channel_adjust = nn.Conv2d(3, out_put_channel*6, kernel_size=1, stride=1)

        #TODO, Replace with a forloop initialization

    self.blocks = nn.ModuleList()
    self.use_single_layer=use_single_layer
    # IF SET TRUE, ERRORS WILL BE THROWN
    if not self.use_single_layer:

        print('________WARNING_______')
        print('This version is not stable')
        print('for handling deeper ConvBlockCSP yet')
        print('BaseConvBlockCSP is the source of the issue, and will be fixed soon')
        print('please set use_single_layer to True')
        print('________WARNING_______')


        for i in range(num_blocks):

                in_channels = input_channel if i == 0 else out_put_channel * (2 * i)
                out_channels = out_put_channel * (2 * (i + 1))

                self.blocks.append(ConvBlockCSP(in_channels, out_put_channel))



  def forward(self, base_feature_map:Tensor)->tuple:
    x1=self.ConvBlockCSP(base_feature_map)

    if(self.use_single_layer):
        if(self.verbose):
          print('x1 part 1',x1[0].shape)
          print('x1 part 2',x1[1].shape)


        return x1

    x1_part1,x1_part2=x1
    if(self.verbose):
        print('x1 part 1',x1_part1.shape)
        print('x1 part 2',x1_part2.shape)

    x2_part1,x2_part2=self.ConvBlockCSP2(x1_part2)
    if(self.verbose):
        print('x2 part 1',x2_part1.shape)
        print('x2 part 2',x2_part2.shape)

    x3_part1, x3_part2=self.ConvBlockCSP3(x2_part2)
    if(self.verbose):
        print('x3 part 1',x3_part1.shape)
        print('x3 part 2',x3_part2.shape)
    x1_part1=self.channel_adjust(x1_part1)
    if(self.verbose):
        print('x1_part1 new shape',x1_part1.shape)

    x1_part1 = nn.functional.adaptive_avg_pool2d(x1_part1, (x3_part2.size(2), x3_part2.size(3)))


    return (x3_part1, torch.cat([x1_part1,x3_part2],1))


class CSPMirrorNet(nn.Module):
   """
   CSPDarkNet:
    unlike the original CSPNet, where we would only take 1 part to pass through convolutions while
    skipping the untoched part to join the output of those convultions, we will be doing the same
    in this architecture with the adddition that we will be joining both both parts in the same
    operation, same model, two different outputs, and concatinate the 2 branches outputs. Hoping to increase the richness of the feature represenation.

    input:
        image: Tensor

    outout:
        Feaute Map:Tensor
   """

   def __init__(self,num_of_base_blocks:int,input_shape:Tensor,verbose:bool=False,overlap_percentage:float=0.20)->None:
      super(CSPMirrorNet,self).__init__()
      self.base_blocks = nn.ModuleList()
      """
      for futur iterations, we will enable depth manupilation of the Mished Dense Block, but keeping in mind with every base_block we will conduct resnetlike operations and adaptive pooling, per proposal 1
      for i in range(num_of_base_blocks):
        self.base_blocks.append(BaseConvBlockCSP(input_shape,4)) #TODO FIX/CHANGEME
      """
      self.base_block = BaseConvBlockCSP(input_shape,4)
      self.verbose=verbose
      self.overlap_percentage=overlap_percentage


   def forward(self, feature_map:Tensor,overlap_percentage:float=0.20)->Tensor: #also can be an image/Feaute map

      if(self.verbose):
        print('before : example size',feature_map.shape)


      height_split = feature_map.shape[2] // 2
      width_split = feature_map.shape[3] // 2
     


      height_overlap = int(height_split * self.overlap_percentage)
      width_overlap = int(width_split * self.overlap_percentage)

      # Using a percentage value for might make the model more flexible by allowing us to scale the overlap relative to the size of the feature map., This is just an idea, not referenced anywhere
      part_1 = feature_map[:, :, :height_split + height_overlap, :width_split + width_overlap]
      part_2 = feature_map[:, :, height_split - height_overlap:, width_split - width_overlap:]

        # Process
      if(self.verbose):
        print("after: ",part_1.shape)
        print("after: ",part_2.shape)


      processed_part1 = self.base_block(part_1)[1]
      processed_part2 = self.base_block(part_2)[1]

      if(self.verbose):
        print("processed_part1", processed_part1.shape)
        print("processed_part2", processed_part2.shape)


      # Reusing the same logic we did earlier, we are mimicking a CSPNet Part1, Part2 except that we are adding cross sectioned, think of it like a siamese network

      example2 = nn.functional.adaptive_avg_pool2d(part_2, (processed_part1.size(2), processed_part1.size(3)))
      example1 = nn.functional.adaptive_avg_pool2d(part_1, (processed_part1.size(2), processed_part1.size(3)))

      concat1 = torch.cat([example2, processed_part1], dim=1)

      concat2 = torch.cat([example1, processed_part2], dim=1)

      # Sum the concatenated outputs, we can also opt out to concatinate them but this will increase computational cost
      combined_output = concat1 + concat2

      if(self.verbose):

        print("Shape of part1 (untouched):", example1.shape)
        print("Shape of example2 (untouched):", example2.shape)
        print("Shape of processed_example1:", processed_part1.shape)
        print("Shape of processed_example2:", processed_part2.shape)
        print("Shape of concat1:", concat1.shape)
        print("Shape of concat2:", concat2.shape)
        print("Combined output shape after summing:", combined_output.shape)


      return combined_output


class FPN(nn.Module):
    """
    Feature Pyramid Network:
    input:
        Feature map: list, outputs from the backbone

    output:
          P1,P2,P3 (each P is a payramid)
    """
    def __init__(self, in_channels: list, out_channels: int):
        super(FPN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels[0], out_channels, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(in_channels[1], out_channels, kernel_size=1, stride=1)
        self.conv3 = nn.Conv2d(in_channels[2], out_channels, kernel_size=1, stride=1)

        self.upconv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1) #maybe
        self.upconv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1) #maybe

    def forward(self, feature_map: list) -> tuple:

        C2, C3, C4 = feature_map

        P3 = self.conv3(C4)
        P2 = self.conv2(C3) + F.interpolate(P3, scale_factor=2, mode='nearest') #Use Bilinear maybe ?
        P1 = self.conv1(C2) + F.interpolate(P2, scale_factor=2, mode='nearest') #Use Bilinear maybe ?

        return P1, P2, P3



class PAN(nn.Module):
  """
  Path Aggregation Network

  """
  def __init__(self):
    super(PAN,self).__init__()
    self.conv1= nn.Conv2d(out_put_channel,out_put_channel,kernel_size=kernel_size,stride=1 )
  def forward(self, P: list)->Tensor:
    P1 , P2, P3 = P

    return None


class APANFPN():
  #We can use permute to reeshape the final layers then padding so we can conduct
  "Attention Path Aggregation Network Feature Pyramid Netork"
  def _init__(self):
    super(APANFPN,self).__init__()
  def forward(self)->None:
    return None


class RPN(nn.Module):
  """
  Region Proposal Network
  """
  def __init__(self):
    super(RPN,self).__init__()
  def forward()->None:
    return None


