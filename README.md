<h1>SmartSwish AI</h1>

 ### [YouTube Demonstration](https://youtu.be/46mltkICIRE)

<h2>Description</h2>
With the automation of the video summarisation process, SmartSwish hopes to provide an easier and more entertaining method to watch NBA games. This project explores the many facets of artificial intelligence (AI) in sports video analysis, the significance of video analysis in sports, and the features and expected advantages of the SmartSwish system.
<br />


<h2>Languages and Tools Used</h2>

- <b>Python</b> 
- <b>YOLOv8</b>
- <b>PyTorch</b> 
- <b>Tensorflow</b>
- <b>VSCode</b>
- <b>Flask</b> 
- <b>HTML & CSS</b>


<h2>Environments Used </h2>

- <b>Windows 10</b> (21H2)

<h2>Project walk-through:</h2>

<p align="center"><br/>
<br />
<body>I used labelImg to go frame by frame on several hundred frames of NBA footage, from different videos, placing boxes over basketball and made baskets. Along with this I experimented with labelling blocks, steals and dunks, but not to a consistent degree. Combining the initial model with this new data helped make the AI model far greater at recognising basketballs in different kinds of lighting and camera angles; even though it still was far from perfect. Using this, I used the model to output all frames where a made basket occurred, forming the basis of a highlight reel.</body>

<br />
To test this feature, I used 3 separate short videos of NBA basketball footage containing different teams and players. The model proved to be moderately effective, since it was able to locate and export highlights of when players scored. However, there were also many false positives that arose from the exported clips, whilst the model struggled with the fact that multiple made basket frames existed next to each other. As a result, I implemented a cool down system so that after the first made basket frame is detected, the system will only show another made basket frame after a certain period.
<br />
From the clips, the system seemed to be able to detect both the basketball and made baskets to a good degree but struggled with blocks, steals and other things. This is due to the experimental nature of using LabelImg for blocks and the system not having enough frames of footage to work with. I was only able to work with a combination of the frames that I input and, unlike with the basketball, I could not utilise transfer learning for this part of the data set. Creating a data set from scratch to recognise blocks and steals proved tedious and did not entail good enough results

</b>
