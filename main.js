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

const UPSCALE_FACTOR = 3
const LATENT_DIM = 100
const NUM_CLASSES = 47
const classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D',
'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'd', 'e', 'f', 'g',
'h', 'n', 'q', 'r', 't']

var generator = (async () => {
    const data = await tf.loadLayersModel('/generator_model/model.json');
    return data;
})();
var canvas = document.getElementById('img-canvas');
var ctx = canvas.getContext("2d")
ctx.fillStyle = 'black'
ctx.fillRect(0, 0, 28, 28)
var btn = document.getElementById('enter-button');
// var input = document.getElementById('input');
var classSliders = generateClassSliders()

btn.addEventListener('click', async event => {
    let g = await generator

    // let text = input.value

    let latentDim = tf.randomNormal([1, 100])
    // let class_ = tf.reshape(tf.oneHot(12, NUM_CLASSES), [1, NUM_CLASSES])
    let class_ = assembleClassVector(classSliders);
    class_.print()
    // latentDim.print()
    //class_.print()

    let img = g.predict([latentDim, class_]).mul(0.5).add(0.5)
    const res = 28*UPSCALE_FACTOR
    img = tf.image.resizeNearestNeighbor(img, [res, res]).reshape([res, res])
    //img.print()

    tf.browser.toPixels(img, canvas)
})