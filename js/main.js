var txt2_items = [
    { file: "camera.mp4", prompt: "Vintage camera with leather case." },
    { file: "house.mp4", prompt: "Two-story brick house with red roof and fence." },
    { file: "orb_pedestal.mp4", prompt: "Glowing orb on a stone pedestal." },
    { file: "spherical_robot.mp4", prompt: "Spherical robot with gold and silver design." },
    { file: "owl.mp4", prompt: "Bronze owl sculpture perched on a branch." },
    { file: "robot_arm.mp4", prompt: "Futuristic robotic arm on a table." },
    { file: "log_cabin.mp4", prompt: "A rustic log cabin with a stone chimney and a wooden porch." },
    { file: "blocky_robot.mp4", prompt: "Blocky, orange and teal robot with articulated limbs." },
    { file: "blaster.mp4", prompt: "Futuristic red toy blaster with transparent magazine." },
    { file: "robot_dog.mp4", prompt: "Metallic dog-like robot with articulated legs and futuristic design elements." },
    { file: "radio.mp4", prompt: "Portable transistor radio, dark cover, speaker grille, brand logo on front." },
    { file: "bulldozer.mp4", prompt: "Yellow and black bulldozer with movable front blade." },
    { file: "ship.mp4", prompt: "Ship with copper and brown hues, intricate deck details." },
    { file: "rocket.mp4", prompt: "A stylized, cartoonish rocket with a red dome top and black antenna, teal cylindrical middle section with red bands and black connectors." },
    { file: "van.mp4", prompt: "A weather-worn vintage delivery van with a boxy shape, a rusted faded green finish, square windows, rusty roof rack." },
    { file: "mansion.mp4", prompt: "A Victorian mansion made of stone bricks with ornate trim, bay windows, and a wraparound porch." },
    { file: "bookshelf.mp4", prompt: "A wooden bookshelf with carved details and adjustable shelves." },
    { file: "chair.mp4", prompt: "A wooden rocking chair with a woven seat and back." },
    { file: "monitor.mp4", prompt: "Vintage green computer monitor." },
    { file: "toy_gun.mp4", prompt: "Sci-fi inspired silver and blue toy gun with intricate design." },
    { file: "tree.mp4", prompt: "The tree has stylized, rounded canopies made up of layered, scale-like leaves in shades of green. Its trunk is twisted." },
    { file: "carriage.mp4", prompt: "The train carriage has a classic, vintage design with a dark, rounded roof, teal exterior, detailed windows, and red wheels." },
    { file: "chess.mp4", prompt: "Carved wooden chess piece. (queen)" },
    { file: "mug.mp4", prompt: "Ceramic mug with a crack." },
    { file: "suitcase.mp4", prompt: "Dark leather suitcase with brass latches." },
    { file: "sculpture.mp4", prompt: "Geometric metal sculpture with angular edges." },
    { file: "lantern.mp4", prompt: "Rustic lantern with a flickering flame." },
];


var img2_items = [
    { file: "trellis.mp4", prompt: "trellis.png" },
    { file: "paper_machine.mp4", prompt: "paper_machine.png" },
    { file: "ship.mp4", prompt: "ship.png" },
    { file: "food_cart.mp4", prompt: "food_cart.png" },
    { file: "loong.mp4", prompt: "loong.png" },
    { file: "mech.mp4", prompt: "mech.png" },
    { file: "castle.mp4", prompt: "castle.png" },
    { file: "toolbox.mp4", prompt: "toolbox.png" },
    { file: "rack.mp4", prompt: "rack.png" },
    { file: "robot_police.mp4", prompt: "robot_police.png" },
    { file: "robot_crab.mp4", prompt: "robot_crab.png" },
    { file: "cart.mp4", prompt: "cart.png" },
    { file: "colorful_cottage.mp4", prompt: "colorful_cottage.png" },
    { file: "house.mp4", prompt: "house.png" },
    { file: "space_colony.mp4", prompt: "space_colony.png" },
    { file: "space_station.mp4", prompt: "space_station.png" },
    { file: "temple.mp4", prompt: "temple.png" },
    { file: "mushroom.mp4", prompt: "mushroom.png" },
    { file: "maya.mp4", prompt: "maya.png" },
    { file: "monkey_astronaut.mp4", prompt: "monkey_astronaut.png" },
    { file: "goblin.mp4", prompt: "goblin.png" },
    { file: "dragonborn.mp4", prompt: "dragonborn.png" },
    { file: "anima_girl.mp4", prompt: "anima_girl.png" },
    { file: "gate.mp4", prompt: "gate.png" },
    { file: "biplane.mp4", prompt: "biplane.png" },
    { file: "elephant.mp4", prompt: "elephant.png" },
    { file: "furry.mp4", prompt: "furry.png" },
    { file: "workbench.mp4", prompt: "workbench.png" },
    { file: "bulldozer.mp4", prompt: "bulldozer.png" },
    { file: "castle2.mp4", prompt: "castle2.png" },
    { file: "chest.mp4", prompt: "chest.png" },
    { file: "excavator.mp4", prompt: "excavator.png" },
    { file: "monster.mp4", prompt: "monster.png" },
    { file: "pickup.mp4", prompt: "pickup.png" },
    { file: "plant.mp4", prompt: "plant.png" },
    { file: "pumpkin.mp4", prompt: "pumpkin.png" },
    { file: "crab_claw.mp4", prompt: "crab_claw.png" },
    { file: "umbrella.mp4", prompt: "umbrella.png" },
    { file: "bench.mp4", prompt: "bench.png" },
    { file: "statue.mp4", prompt: "statue.png" },
    { file: "trunk.mp4", prompt: "trunk.png" },
    { file: "ceramic_elephant.mp4", prompt: "ceramic_elephant.png" },
    { file: "light.mp4", prompt: "light.png" },
    { file: "microphone.mp4", prompt: "microphone.png" },
    { file: "sofa.mp4", prompt: "sofa.png" },
    { file: "sofa_cat.mp4", prompt: "sofa_cat.png" },
    { file: "bomb.mp4", prompt: "bomb.png" },
    { file: "car.mp4", prompt: "car.png" },
];


function txt2_carousel_item_template(item) {
    return `<div class="x-card">
                <div style="width: 100%; aspect-ratio: 1">
                    <video autoplay loop muted height="100%" src="/assets/txt2/${item.file}"></video>
                </div>
                <div class="caption x-handwriting">${item.prompt}</div>
            </div>`;
}

function img2_carousel_item_template(item) {
    return `<div class="x-card">
                <div style="width: 100%; aspect-ratio: 1">
                    <video autoplay loop muted height="100%" src="/assets/img2/${item.file}"></video>
                </div>
                <div class="caption">
                    <img src="/assets/img2/${item.prompt}" height="100%" style="border: 1px solid black;">
                </div>
            </div>`;
}


function supportsHEVC() {
    var video = document.createElement('video');
    return video.canPlayType('video/mp4; codecs="hev1.1.6.L93.B0"') !== '';
}


window.onload = function() {
    make_carousel('results-txt2', txt2_carousel_item_template, txt2_items, 3);
    make_carousel('results-img2', img2_carousel_item_template, img2_items, 3);
};