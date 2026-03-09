let video = document.getElementById("video")

function startCamera(){

navigator.mediaDevices.getUserMedia({video:true})
.then(stream=>{
video.srcObject = stream
})

}
function startAutoScan(){

setInterval(()=>{

detect()

},2000)

}

function detect(){

let canvas = document.createElement("canvas")
canvas.width = 320
canvas.height = 240

let ctx = canvas.getContext("2d")
ctx.drawImage(video,0,0)

let image = canvas.toDataURL("image/jpeg")

fetch("/detect",{

method:"POST",

body:JSON.stringify({
image:image
}),

headers:{
"Content-Type":"application/json"
}

})
.then(res=>res.json())
.then(data=>{

document.getElementById("result").innerText =
"Detection Result : " + data.result

})

}