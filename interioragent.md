---
layout: default
title: "Nerfify: Multi-Agent in Practice"
nav_order: 10
---

# InteriorAgent: A Template for building Agentic Generative AI

<br>

Now that we have learned the basics of an LLM Agent and how to build one, let's see what it takes to actually build something for a real problem. We will learn about the case study of my recent 3DV 2026 paper **InteriorAgent: LLM Agent for Interior Design-Aware 3D Layout Generation**. I believe this project is a very good case study in designing agentic systems because like Agent252D, I made it from scratch! without utilizing pre existing LLM frameworks (because at the time I began working on this project, there were none that worked well enough). We will be exploring several key concepts such as designing domain effective representations, tools, program synthesis as well as optimization and most importantly *context engineering*. So let's get started!

## Problem

Consider the task of creating an interior layout given a user prompt such as *`a bedroom`* along with an optional reference image. We want to create a system that can regress the 4D coordinates (x,y,z, orientation) of each asset to be placed within the scene. For simplicity, assume that the assets are already retrieved from a large repository of indoor assets. 
<br>
<p style="text-align: center;">
  <img src="assets/layout_intro.png" alt="interioragentpipeline" style="width: 90%;">
</p>
<br>

Creating interior layout designs has numerous applications, including virtual reality, architectural visualization and real estate planning. Generating realistic and functional indoor scenes requires a nuanced understanding of
spatial configurations and human-centered design principles. 

Conventional wisdom suggests that we fit the largest possible model on the largest available dataset containing pairs of (prompt, layout). 

<br>
<p style="text-align: center;">
  <img src="assets/conventional_wisdom.png" alt="interioragentpipeline" style="width: 90%;">
</p>
<br>

Unfortunately, this approach has several limitations: 

1. Finding such large enough datasets is often difficult, most problems in 3D Computer vision are affected from this curse. 
2. Extending the capabilities (like adding humans into the scene) requires manufacturing a new dataset and training from scratch.
3. Cannot easily train for editing functionality unless have paired dataset for that too! (extremely rare).
4. Doesn't have world knowledge like cannot create a bedroom layout based on *latest trends in 2026*. 

## Solution

We can leverage pretrained LLMs like GPT-5 to help us compute the pose of all the assets. Naively, this entails feeding the input prompt along with asset information (like name, size, images, etc) to the LLM and asking it to output the final asset poses. 

<br>
<p style="text-align: center;">
  <img src="assets/naive_llm.png" alt="interioragentpipeline" style="width: 90%;">
</p>
<br>

This sounds feasible, however, we find that this doesn't work well in practise. This is because, LLMs which are trained for discrete symbolic tasks in NLP such as translation, coding, etc do not work equally well for continuous, low-level tasks such as regression which is the case in most computer vision problems. Similar to the interior layout, we can imagine regressing 6DOF bounding boxes (3D object detection) from single images, regressing per pixel depth from single image using LLMs. Doesn't sound very nice does it? Not all tasks in computer vision can be replaced with an LLM, however as we shall see, for a majority of existing problems (and potentially innumerable newer problems now solvevable) we can orchestrate LLM based agentic workflows together with tradtional computer vision pipelines to build better and more capable systems. 

Coming back to the problem of interior design. How do we get more accurate layouts from our LLM?
The answer is *domain specific api/language*. Consider the following program.

```python
from IDSDL.scene import SceneProgRoom
scene = SceneProgRoom("simple_bed_room")

bed = scene.AddAsset("a queen-sized bed with a wooden frame and a plush mattress")
nightstand = scene.AddAsset("a small wooden nightstand with a drawer")
lamp = scene.AddAsset("a modern table lamp with a white shade")
rug = scene.AddAsset("a soft neutral area rug")
cabinet = scene.AddAsset("a tall and wide wooden wardrobe with mirrored doors")
nightstand.place_on_top(lamp)
bed.place_on_back_left(nightstand)
bed.place_on_back_right(nightstand)
bed.place_under(rug)
bed.place_on_right_further(cabinet)
```

Note that the layout is expressed as a program where the relations between the assets are specified instead of directly listintng the respective pose. When the program executes, it automatically computes the resultant poses. This is because, underlying each functionality, there is dedicated program which carries out low level execution and this way provides an intuitive abstraction for the LLM to focus on specifying the high level structure of the layout and leaving the low level pose computation to grounded operations via tools or established models in interior design.
```python
@placemethod
def place_on_back_right(self, obj):
    front_dir, back_dir, left_dir, right_dir, center, width, height, depth = self.get_anchor_center_dirs()
    back_right = center + back_dir * (depth / 2 - obj.get_depth() / 2) + right_dir * (width / 2 + obj.get_width() / 2 + SIDE_GAP)
    obj.set_location(back_right[0], self.compute_obj_y(obj), back_right[2])
    obj.set_rotation(0)
    self.add_child(obj)
```
As a principle, when it comes to solving computer vision problems via LLM Agents, we need to give a great degree of consideration to building these abstractions in order to form our problem specific *api/language* which will be then used by the LLM to express a response. 
InteriorAgent uses a sophesticated Interior Design aware Scene Description Language to express a variety of interior design layouts. The following figure shows an example program for creating *`a living room with a toy pony`* 

<br>
<p style="text-align: center;">
  <img src="assets/program_in_IDSDL.png" alt="interioragentpipeline" style="width: 90%;">
</p>
<br>

Note the flexibility provided by the language in order to create a layout without burdening the LLM with regressing the pose for various objects. The IDSDL is so powerful not only because of its core interior-design based logic for placement, but because it also leverages several domain specific tools to solve niche problems such as creating paintings using Stable Diffusion. 


1. show how given an input image, the idea is to get 3D quantities. (classical computer vision used to do this low level stuff). 
2. Need for an intermediate representation that connects language to low level representation.
3. Show a few examples of intermediate representations (programs) for various computer vision tasks. 
4. What makes a good representation? detailed documentation, hopefully lots of demonstrations
5. 