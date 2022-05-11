function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

function normalizeTensor(data) {
    const dataLength = data.square().sum().sqrt()
    return data.div(dataLength);
}

function generateClassSliders() {
    let container = document.getElementById('classSlidersContainer');
    let sliders = [];
    for (let i = 0; i < NUM_CLASSES; i++) {
        span = document.createElement('span')
        content = document.createTextNode(classes[i])
        span.appendChild(content)
        container.appendChild(span)
        let newSlider = container.appendChild(document.createElement('input'))
        newSlider.type = 'range'
        newSlider.min = 0,
        newSlider.max = 2000,
        newSlider.value = 0,
        newSlider.id = "slider_" + i.toString()
        newSlider.addEventListener("pointermove", generateImage)
        sliders.push(newSlider)
        container.appendChild(document.createElement('br'))
    }
    return sliders
}

function assembleClassVector(sliders, normalize=false) {
    classArr = []
    sliders.forEach(slider => {
        classArr.push(slider.value / 1000)
    });
    let classTensor = tf.tensor1d(classArr)
    if (normalize) {
        classTensor = normalizeTensor(classTensor)
    }
    classTensor = classTensor.reshape([1, NUM_CLASSES])
    return classTensor
}


mouseDown = false
leftButton = false
rightButton = false
mouseX = -1
mouseY = -1

class MouseHandler {
    constructor(object) {
        object.addEventListener('pointermove', this.pointerMove)
        object.addEventListener('pointerdown', this.pointerDown)
        object.addEventListener('pointerup', this.pointerUp)
        object.addEventListener('pointerout', this.pointerUp)
        mouseDown = false
        leftButton = false
        rightButton = false
        mouseX = -1
        mouseY = -1
    }

    async pointerMove(event) {
        mouseX = event.offsetX
        mouseY = event.offsetY
        // event.preventDefault()
    }

    async pointerDown(event) {
        mouseDown = true
        mouseX = event.offsetX
        mouseY = event.offsetY
        leftButton = (event.buttons & 0b01) || (event.buttons == 0)
        rightButton = event.buttons & 0b10
        // event.preventDefault()
    }

    async pointerUp(event) {
        mouseDown = false
        mouseX = -1
        mouseY = -1
        leftButton = false
        rightButton = false
        // event.preventDefault()
    }
}

drawValue = 1
let drawPos = document.getElementById('draw-color-positive')
drawPos.addEventListener('change', () => {drawValue = 1})
let drawNeg = document.getElementById('draw-color-negative')
drawNeg.addEventListener('change', () => {drawValue = -1})

class LatentVecHandler extends MouseHandler {
    constructor(canvas, pixel=10) {
        super(canvas)

        this.canvas = canvas
        this.ctx = canvas.getContext("2d")

        this.pixel = pixel
        this.changeAmount = 0.05
        
        this.valueRange = 2.0
        this.minValue = -this.valueRange
        this.maxValue = this.valueRange
        
        this.randomize()
    }

    randomize() {
        let arr = tf.randomNormal([10, 10])
        arr = arr.sub(arr.min())
        arr = arr.div(arr.max()).add(-0.5).mul(2).mul(this.valueRange)
        this.array = arr.arraySync()
        this.redraw()
    }

    async mainLoop() {
        while (true) {
            if (mouseDown) {
                let row = Math.floor(mouseY / this.pixel)
                let col = Math.floor(mouseX / this.pixel)
                let change = this.changeAmount * (leftButton - rightButton)
                let newValue = this.array[row][col] + change * drawValue
                console.log(drawValue)
                if (newValue > this.minValue && newValue < this.maxValue)
                    this.array[row][col] = newValue;
                // if (newValue)
                this.redraw()
                await generateImage()
            }
            await sleep(10)
        }
    }

    colorOf(value) {
        if (value < this.minValue) {value = this.minValue};
        if (value > this.maxValue) {value = this.maxValue};
        let red = 0
        let blue = 0
        let saturation = Math.floor(Math.abs(value) / this.maxValue * 2 * 255)
        if (value > 0) {
            blue = saturation
        } else {
            red = saturation
        }
        return `rgb(${red}, 0, ${blue})`
    }

    redraw() {
        for (let row = 0; row < 10; row++){
            for (let col = 0; col < 10; col++){
                let value = this.array[row][col]
                this.ctx.fillStyle = this.colorOf(value)
                this.ctx.fillRect(col * this.pixel, row * this.pixel, this.pixel, this.pixel)
            }
        }
    }

    getLatentVec() {
        return tf.tensor2d(this.array).reshape([1, 100])
    }
}

function fillCanvas(canvas, x, y, color) {
    let ctx = canvas.getContext("2d")
    ctx.fillStyle = color
    ctx.fillRect(0, 0, x, y)
}

function disableRightClick(object) {
    object.oncontextmenu = function(e) { e.preventDefault(); e.stopPropagation(); }
}

async function generateImage(event) {
    let g = await generator

    // let text = input.value

    // let latentVec = tf.randomNormal([1, 100])
    let latentVec = latentHandler.getLatentVec()
    // let class_ = tf.reshape(tf.oneHot(12, NUM_CLASSES), [1, NUM_CLASSES])
    let class_ = assembleClassVector(classSliders);
    // class_.print()
    // latentDim.print()
    //class_.print()

    let img = g.predict([latentVec, class_]).mul(0.5).add(0.5)
    img = tf.image.resizeNearestNeighbor(img, [res, res]).reshape([res, res])
    //img.print()

    tf.browser.toPixels(img, resultCanvas)
}

const UPSCALE_FACTOR = 7
const res = 28 * UPSCALE_FACTOR
const LATENT_DIM = 100
const LATENT_PIXEL_SIZE = 20
const NUM_CLASSES = 47
const classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D',
'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'd', 'e', 'f', 'g',
'h', 'n', 'q', 'r', 't']

var generator = (async () => {
    const data = await tf.loadLayersModel('/generator_model/model.json');
    return data;
})();

var resultCanvas = document.getElementById('img-canvas');
resultCanvas.width = res
resultCanvas.height = res
fillCanvas(resultCanvas, res, res, 'black')

var latentCanvas = document.getElementById('latent-vector-canvas');
fillCanvas(latentCanvas, LATENT_PIXEL_SIZE * 10, LATENT_PIXEL_SIZE * 10, 'black')
latentCanvas.width = LATENT_PIXEL_SIZE * 10
latentCanvas.height = LATENT_PIXEL_SIZE * 10
disableRightClick(latentCanvas)
var latentHandler = new LatentVecHandler(latentCanvas, LATENT_PIXEL_SIZE)
const ml = (async () => {
    const data = await latentHandler.mainLoop();
    return data;
})();

let randomizeLatentBtn = document.getElementById('randomize-latent-btn')
randomizeLatentBtn.addEventListener('click', async e => {
    latentHandler.randomize()
    await generateImage()
})

// var input = document.getElementById('input');
var classSliders = generateClassSliders()
