Prerequisite
(i)use python 3.11.x
(ii)install uv
(iii)in each directory, issue "uv sync"


then 

(I)
for example, you put your code in in C:\xxx\yyy\changedetection\webapi
in C:\xxx\yyy\changedetection\webapi, "uv run .\app.py"


(II)
for example, you put your code in in C:\xxx\yyy\changedetection\scrap-graph
in C:\xxx\yyy\changedetection\scrap-graph, "uv run .\worker.py"

(III)

for example, you put your code in in C:\xxx\yyy\changedetection\changedetect
(a) create directory "news_monitor" in C:\xxx\yyy\changedetection\
(b) in C:\xxx\yyy\changedetection\changedetect", issue "uv tool run changedetection.io -d C:\Users\admin\Documents\aliya\changedetection\news_monitor -p 5000"

 

