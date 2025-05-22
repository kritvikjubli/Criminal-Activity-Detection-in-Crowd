let nextButton = document.getElementById('next');
let prevButton = document.getElementById('prev');
let carousel = document.querySelector('.carousel');
let listHTML = document.querySelector('.carousel .list');
let seeMoreButtons = document.querySelectorAll('.seeMore');
let backButton = document.getElementById('back');

nextButton.onclick = function(){
    showSlider('next');
}
prevButton.onclick = function(){
    showSlider('prev');
}
let unAcceppClick;
const showSlider = (type) => {
    nextButton.style.pointerEvents = 'none';
    prevButton.style.pointerEvents = 'none';

    carousel.classList.remove('next', 'prev');
    let items = document.querySelectorAll('.carousel .list .item');
    if(type === 'next'){
        listHTML.appendChild(items[0]);
        carousel.classList.add('next');
    }else{
        listHTML.prepend(items[items.length - 1]);
        carousel.classList.add('prev');
    }
    clearTimeout(unAcceppClick);
    unAcceppClick = setTimeout(()=>{
        nextButton.style.pointerEvents = 'auto';
        prevButton.style.pointerEvents = 'auto';
    }, 2000)
}
seeMoreButtons.forEach((button) => {
    button.onclick = function(){
        carousel.classList.remove('next', 'prev');
        carousel.classList.add('showDetail');
    }
});
backButton.onclick = function(){
    carousel.classList.remove('showDetail');
}
// let i = document.querySelectorAll('.i');
// console.log(i);
// i.forEach(i => {
//     i.addEventListener('mousemove', (e)=>{
//         let positionPx = e.x - i.getBoundingClientRect().left;
//         let positionX = (positionPx / i.offsetWidth) * 100;
//         console.log(positionX, positionPx);

//         let positionPy = event.y - i.getBoundingClientRect().top;
//         let positionY = (positionPy / i.offsetHeight) * 100;

        
//         i.style.setProperty('--rX', (0.5)*(50 - positionY) + 'deg');
//         i.style.setProperty('--rY', -(0.5)*(50 - positionX) + 'deg');
//     })
//     i.addEventListener('mouseout', ()=>{
//         i.style.setProperty('--rX', '0deg');
//         i.style.setProperty('--rY', '0deg');
//     })
// })

document.getElementById("start-video").addEventListener("click", function() {
    let video = document.getElementById("crime-video");
    video.classList.remove("blur-md");
    this.style.display = "none";  
    video.play();
});
