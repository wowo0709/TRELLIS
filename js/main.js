var txt2_items = [
    { video: "camera.mp4", prompt: "Vintage camera with leather case." },
    { video: "house.mp4", prompt: "Two-story brick house with red roof and fence." },
    { video: "orb_pedestal.mp4", prompt: "Glowing orb on a stone pedestal." },
    { video: "spherical_robot.mp4", prompt: "Spherical robot with gold and silver design." },
    { video: "owl.mp4", prompt: "Bronze owl sculpture perched on a branch." },
    { video: "robot_arm.mp4", prompt: "Futuristic robotic arm on a table." },
    { video: "log_cabin.mp4", prompt: "A rustic log cabin with a stone chimney and a wooden porch." },
    { video: "blocky_robot.mp4", prompt: "Blocky, orange and teal robot with articulated limbs." },
    { video: "blaster.mp4", prompt: "Futuristic red toy blaster with transparent magazine." },
    { video: "robot_dog.mp4", prompt: "Metallic dog-like robot with articulated legs and futuristic design elements." },
    { video: "radio.mp4", prompt: "Portable transistor radio, dark cover, speaker grille, brand logo on front." },
    { video: "bulldozer.mp4", prompt: "Yellow and black bulldozer with movable front blade." },
    { video: "ship.mp4", prompt: "Ship with copper and brown hues, intricate deck details." },
    { video: "rocket.mp4", prompt: "A stylized, cartoonish rocket with a red dome top and black antenna, teal cylindrical middle section with red bands and black connectors." },
    { video: "van.mp4", prompt: "A weather-worn vintage delivery van with a boxy shape, a rusted faded green finish, square windows, rusty roof rack." },
    { video: "mansion.mp4", prompt: "A Victorian mansion made of stone bricks with ornate trim, bay windows, and a wraparound porch." },
    { video: "bookshelf.mp4", prompt: "A wooden bookshelf with carved details and adjustable shelves." },
    { video: "chair.mp4", prompt: "A wooden rocking chair with a woven seat and back." },
    { video: "monitor.mp4", prompt: "Vintage green computer monitor." },
    { video: "toy_gun.mp4", prompt: "Sci-fi inspired silver and blue toy gun with intricate design." },
    { video: "tree.mp4", prompt: "The tree has stylized, rounded canopies made up of layered, scale-like leaves in shades of green. Its trunk is twisted." },
    { video: "carriage.mp4", prompt: "The train carriage has a classic, vintage design with a dark, rounded roof, teal exterior, detailed windows, and red wheels." },
    { video: "chess.mp4", prompt: "Carved wooden chess piece. (queen)" },
    { video: "mug.mp4", prompt: "Ceramic mug with a crack." },
    { video: "suitcase.mp4", prompt: "Dark leather suitcase with brass latches." },
    { video: "sculpture.mp4", prompt: "Geometric metal sculpture with angular edges." },
    { video: "lantern.mp4", prompt: "Rustic lantern with a flickering flame." },
];


var img2_items = [
    { video: "trellis.mp4", prompt: "trellis.png" },
    { video: "paper_machine.mp4", prompt: "paper_machine.png" },
    { video: "ship.mp4", prompt: "ship.png" },
    { video: "food_cart.mp4", prompt: "food_cart.png" },
    { video: "loong.mp4", prompt: "loong.png" },
    { video: "mech.mp4", prompt: "mech.png" },
    { video: "castle.mp4", prompt: "castle.png" },
    { video: "toolbox.mp4", prompt: "toolbox.png" },
    { video: "rack.mp4", prompt: "rack.png" },
    { video: "robot_police.mp4", prompt: "robot_police.png" },
    { video: "robot_crab.mp4", prompt: "robot_crab.png" },
    { video: "cart.mp4", prompt: "cart.png" },
    { video: "colorful_cottage.mp4", prompt: "colorful_cottage.png" },
    { video: "house.mp4", prompt: "house.png" },
    { video: "space_colony.mp4", prompt: "space_colony.png" },
    { video: "space_station.mp4", prompt: "space_station.png" },
    { video: "temple.mp4", prompt: "temple.png" },
    { video: "mushroom.mp4", prompt: "mushroom.png" },
    { video: "maya.mp4", prompt: "maya.png" },
    { video: "monkey_astronaut.mp4", prompt: "monkey_astronaut.png" },
    { video: "goblin.mp4", prompt: "goblin.png" },
    { video: "dragonborn.mp4", prompt: "dragonborn.png" },
    { video: "anima_girl.mp4", prompt: "anima_girl.png" },
    { video: "gate.mp4", prompt: "gate.png" },
    { video: "biplane.mp4", prompt: "biplane.png" },
    { video: "elephant.mp4", prompt: "elephant.png" },
    { video: "furry.mp4", prompt: "furry.png" },
    { video: "workbench.mp4", prompt: "workbench.png" },
    { video: "bulldozer.mp4", prompt: "bulldozer.png" },
    { video: "castle2.mp4", prompt: "castle2.png" },
    { video: "chest.mp4", prompt: "chest.png" },
    { video: "excavator.mp4", prompt: "excavator.png" },
    { video: "monster.mp4", prompt: "monster.png" },
    { video: "pickup.mp4", prompt: "pickup.png" },
    { video: "plant.mp4", prompt: "plant.png" },
    { video: "pumpkin.mp4", prompt: "pumpkin.png" },
    { video: "crab_claw.mp4", prompt: "crab_claw.png" },
    { video: "umbrella.mp4", prompt: "umbrella.png" },
    { video: "bench.mp4", prompt: "bench.png" },
    { video: "statue.mp4", prompt: "statue.png" },
    { video: "trunk.mp4", prompt: "trunk.png" },
    { video: "ceramic_elephant.mp4", prompt: "ceramic_elephant.png" },
    { video: "light.mp4", prompt: "light.png" },
    { video: "microphone.mp4", prompt: "microphone.png" },
    { video: "sofa.mp4", prompt: "sofa.png" },
    { video: "sofa_cat.mp4", prompt: "sofa_cat.png" },
    { video: "bomb.mp4", prompt: "bomb.png" },
    { video: "car.mp4", prompt: "car.png" },
];


function supportsHEVC() {
    var video = document.createElement('video');
    return video.canPlayType('video/mp4; codecs="hev1.1.6.L93.B0"') !== '';
}


function txt2_carousel_item_template(item) {
    return `<div class="x-card" onclick=\'openWindow(txt2_window_template(${JSON.stringify(item)}))\'>
                <div style="width: 100%; aspect-ratio: 1">
                    <video autoplay loop muted height="100%" src="/assets/txt2/videos/${item.video}"></video>
                </div>
                <div class="caption x-handwriting">${item.prompt}</div>
            </div>`;
}

function img2_carousel_item_template(item) {
    return `<div class="x-card">
                <div style="width: 100%; aspect-ratio: 1">
                    <video autoplay loop muted height="100%" src="/assets/img2/videos/${item.video}"></video>
                </div>
                <div class="caption">
                    <img src="/assets/img2/images/${item.prompt}" height="100%" style="border: 1px solid black;">
                </div>
            </div>`;
}


function txt2_window_template(item) {
    return `<div class="x-row">
                <div class="modelviewer-container">
                    <model-viewer
                        src="/assets/scenes/blacksmith/glbs_hq/paper_machine.glb"
                        camera-controls
                        tone-mapping="neutral"
                        shadow-intensity="1"
                        environment-image="/assets/white.jpg"
                        exposure="5">
                    </model-viewer>
                </div>
                <div class="modelviewer-panel">
                    <div class="caption x-handwriting">${item.prompt}</div>
                </div>
            </div>`;
}


function openWindow(content) {
    let fullscreenElement = document.getElementById('fullscreen');
    let contentElement = fullscreenElement.querySelector('#content');
    contentElement.innerHTML = content;
    fullscreenElement.style.display = 'flex';
    setTimeout(() => {
        fullscreenElement.style.opacity = 1;
    }, 100);
}


function closeWindow() {
    let fullscreenElement = document.getElementById('fullscreen');
    fullscreenElement.style.opacity = 0;
    setTimeout(() => {
        fullscreenElement.style.display = 'none';
    }, 500);
}


window.onload = function() {
    make_carousel('results-txt2', txt2_carousel_item_template, txt2_items, 3);
    make_carousel('results-img2', img2_carousel_item_template, img2_items, 3);
};