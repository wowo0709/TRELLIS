var scene_items = [
    {
        title: "Vibrant Street View",
        video: "street/street.mp4",
        layout: "street/layout.json",
        model: "street/street.glb",
    },
    {
        title: "Dwarf Blacksmith Shop",
        video: "blacksmith/blacksmith.mp4",
        layout: "blacksmith/layout.json",
        model: "blacksmith/blacksmith.glb",
    },
];

async function prepare_scene_items() {
    for (let i=0; i<scene_items.length; i++) {
        let item = scene_items[i];
        let layout = await fetch('/assets/scenes/' + item.layout);
        item.layout = await layout.json();
        item.hotspots = [];
        for (let j=0; j<item.layout.assets.length; j++) {
            let asset = item.layout.assets[j];
            if (item.hotspots.find(hotspot => hotspot.name === asset.name)) {
                continue;
            }
            item.hotspots.push({
                name: asset.name,
                position: [asset.transform.location.x, asset.transform.location.z, -asset.transform.location.y],
            });
        }
    }
}

function scene_carousel_item_template(item) {
    return `<div class="x-card">
                <div class="x-row" style="margin-bottom: 16px">
                    <div style="font-size: 28px; font-weight: 700; margin-left: 8px">${item.title}</div>
                    <div class="x-flex-spacer"></div>
                    <div class="x-button" onclick=\'openWindow(scene_window_template(${JSON.stringify(item)}))\'>View GLB</div>
                </div>
                <div style="width: 100%; aspect-ratio: 16/9">
                    <video controls muted height="100%" src="/assets/scenes/${item.video}"></video>
                </div>
            </div>`;
}

function scene_window_template(item) {
    function scene_panel_template(item) {
        return `
            <div style="font-size: 28px; font-weight: 700; margin: 8px 0px 0px 4px">${item.title}</div>
            <div class="x-section-title small"><div class="x-gradient-font">Display Mode</div></div>
            <div class="x-left-align">
                <div id="appearance-button" class="modelviewer-panel-button small checked" onclick="showTexture()">Appearance</div>
                <div id="geometry-button" class="modelviewer-panel-button small" onclick="hideTexture()">Geometry</div>
            </div>
            <div class="x-section-title small"><div class="x-gradient-font">Assets</div></div>
            <div class="x-left-align" style="flex-wrap: wrap">
                ${item.hotspots.map(hotspot => `
                    <div class="modelviewer-panel-button small checked">${hotspot.name}</div>
                `).join('')}
            </div>
            <div class="x-flex-spacer"></div>
            <div class="x-row">
                <div id="download-button" class="modelviewer-panel-button checked" onclick="window.open('${item.model}')">Download GLB</div>
            </div>
        `;
    }
    item = JSON.parse(JSON.stringify(item));
    item.model = '/assets/scenes/' + item.model
    return modelviewer_window_template(item, scene_panel_template, {viewer_size: 750, panel_size: 350, show_hotspots: true});
}
