const CANVAS_WIDTH = 960;
const CANVAS_HEIGHT = 540;
const GRAVITY = 1800;
const FLOOR_Y = 432;
const WORLD_LEFT = 48;
const WORLD_RIGHT = 912;
const PLAYER_SCALE = 3;
const DEBUG_RENDER = new URLSearchParams(window.location.search).has("debug");

const canvas = document.querySelector("#game");
const ctx = canvas.getContext("2d");
const statusNode = document.querySelector("#status");

ctx.imageSmoothingEnabled = false;

const keys = new Set();

window.addEventListener("keydown", (event) => {
  if (["Space", "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"].includes(event.code)) {
    event.preventDefault();
  }
  keys.add(event.code);
});

window.addEventListener("keyup", (event) => {
  keys.delete(event.code);
});

const loadJson = async (path) => {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`Failed to load ${path}: ${response.status}`);
  }
  return response.json();
};

const loadImage = (path) =>
  new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error(`Failed to load image: ${path}`));
    image.src = path;
  });

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function animationByLabel(entityAnimations, label, fallback = null) {
  if (entityAnimations?.[label]) {
    return entityAnimations[label];
  }
  if (fallback && entityAnimations?.[fallback]) {
    return entityAnimations[fallback];
  }
  return null;
}

function animationDurationMs(animation) {
  if (!animation?.frames?.length) {
    return 0;
  }
  if (Array.isArray(animation.frame_durations) && animation.frame_durations.length === animation.frames.length) {
    return animation.frame_durations.reduce((sum, duration) => sum + duration, 0);
  }
  return animation.frames.length * (animation.frame_duration ?? 100);
}

function frameAtTime(animation, elapsedMs, loop = true) {
  if (!animation?.frames?.length) {
    return null;
  }

  const totalDuration = animationDurationMs(animation);
  if (totalDuration <= 0) {
    return animation.frames[0];
  }

  let clock = elapsedMs;
  if (loop) {
    clock %= totalDuration;
  } else {
    clock = clamp(clock, 0, Math.max(0, totalDuration - 1));
  }

  if (Array.isArray(animation.frame_durations) && animation.frame_durations.length === animation.frames.length) {
    let accumulated = 0;
    for (let index = 0; index < animation.frames.length; index += 1) {
      accumulated += animation.frame_durations[index];
      if (clock < accumulated) {
        return animation.frames[index];
      }
    }
    return animation.frames[animation.frames.length - 1];
  }

  const frameDuration = animation.frame_duration ?? 100;
  const frameIndex = Math.min(animation.frames.length - 1, Math.floor(clock / frameDuration));
  return animation.frames[frameIndex];
}

function spriteDrawMetrics(sprite, anchorX, anchorY, scale = 1, flip = false) {
  const anchor = sprite.anchor ?? { x: 0, y: 0 };
  const width = sprite.w * scale;
  const height = sprite.h * scale;
  const x = flip ? anchorX - (sprite.w - anchor.x) * scale : anchorX - anchor.x * scale;
  const y = anchorY - anchor.y * scale;

  return {
    x,
    y,
    w: width,
    h: height,
    anchorX,
    anchorY,
  };
}

function drawDebugOverlay(sprite, metrics) {
  if (!DEBUG_RENDER) {
    return;
  }
  if ((sprite.classification ?? "").includes("background")) {
    return;
  }

  ctx.save();
  ctx.strokeStyle = "#00ff66";
  ctx.lineWidth = 1;
  ctx.strokeRect(metrics.x + 0.5, metrics.y + 0.5, metrics.w, metrics.h);
  ctx.fillStyle = "#ff2a2a";
  ctx.beginPath();
  ctx.arc(metrics.anchorX, metrics.anchorY, 3, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

function drawSprite(atlasImages, sprite, anchorX, anchorY, scale = 1, flip = false, alpha = 1) {
  const atlasImage = atlasImages.get(sprite.atlas);
  if (!atlasImage) {
    return null;
  }
  const anchor = sprite.anchor ?? { x: 0, y: 0 };
  const metrics = spriteDrawMetrics(sprite, anchorX, anchorY, scale, flip);
  ctx.save();
  ctx.globalAlpha = alpha;
  if (flip) {
    ctx.translate(anchorX, anchorY);
    ctx.scale(-1, 1);
    ctx.drawImage(
      atlasImage,
      sprite.x,
      sprite.y,
      sprite.w,
      sprite.h,
      -(sprite.w - anchor.x) * scale,
      -anchor.y * scale,
      metrics.w,
      metrics.h,
    );
  } else {
    ctx.drawImage(
      atlasImage,
      sprite.x,
      sprite.y,
      sprite.w,
      sprite.h,
      metrics.x,
      metrics.y,
      metrics.w,
      metrics.h,
    );
  }
  ctx.restore();
  drawDebugOverlay(sprite, metrics);
  return metrics;
}

function groundTopAt(x) {
  if (x > 592 && x < 756) {
    return 328;
  }
  if (x > 300 && x < 480) {
    return 366;
  }
  return FLOOR_Y;
}

async function boot() {
  const [animations, atlasData] = await Promise.all([
    loadJson("../outputs/metadata/animations.json"),
    loadJson("../outputs/metadata/atlas.json"),
  ]);

  const atlasEntries = atlasData.atlases ?? [];
  const atlasImages = new Map(
    await Promise.all(
      atlasEntries.map(async (entry) => {
        const image = await loadImage(`../outputs/${entry.file}`);
        return [entry.file, image];
      }),
    ),
  );

  const sprites = atlasData.sprites;
  const playerAnimations = animations.player ?? {};
  const auraAnimation = animationByLabel(animations.aura, "effects");
  const beamAnimation = animationByLabel(animations.light_beam, "effects");

  const player = {
    x: 180,
    y: FLOOR_Y,
    vx: 0,
    vy: 0,
    facing: 1,
    animation: "idle",
    animationTime: 0,
    attackTime: 99,
    auraTime: 99,
    beamTime: 99,
    jumpPressed: false,
  };

  const platforms = [
    { x: 32, y: FLOOR_Y, sprite: "platform_2" },
    { x: 248, y: FLOOR_Y, sprite: "platform_2" },
    { x: 464, y: FLOOR_Y, sprite: "platform_2" },
    { x: 680, y: FLOOR_Y, sprite: "platform_2" },
    { x: 304, y: 366, sprite: "platform_5" },
    { x: 600, y: 328, sprite: "platform_7" },
  ];

  const backgroundNames = ["background_0", "background_3", "background_5"];

  let lastTime = performance.now();

  function setAnimation(nextAnimation) {
    if (player.animation !== nextAnimation) {
      player.animation = nextAnimation;
      player.animationTime = 0;
    }
  }

  function update(dt) {
    const left = keys.has("KeyA") || keys.has("ArrowLeft");
    const right = keys.has("KeyD") || keys.has("ArrowRight");
    const running = keys.has("ShiftLeft") || keys.has("ShiftRight");
    const jump = keys.has("Space");

    if (keys.has("KeyZ") && player.attackTime > 0.35) {
      player.attackTime = 0;
    }
    if (keys.has("KeyX") && player.auraTime > 0.8) {
      player.auraTime = 0;
    }
    if (keys.has("KeyC") && player.beamTime > 0.8) {
      player.beamTime = 0;
    }

    const speed = running ? 240 : 148;
    const moveIntent = (right ? 1 : 0) - (left ? 1 : 0);
    const targetVelocity = moveIntent * speed;
    const acceleration = player.y >= groundTopAt(player.x) ? 16 : 8;
    player.vx = lerp(player.vx, targetVelocity, clamp(acceleration * dt, 0, 1));

    if (moveIntent !== 0) {
      player.facing = moveIntent > 0 ? 1 : -1;
    }

    const currentGround = groundTopAt(player.x);
    const onGround = player.y >= currentGround - 1 && player.vy >= 0;

    if (jump && !player.jumpPressed && onGround) {
      player.vy = -690;
    }
    player.jumpPressed = jump;

    player.vy += GRAVITY * dt;
    player.x += player.vx * dt;
    player.y += player.vy * dt;
    player.x = clamp(player.x, WORLD_LEFT, WORLD_RIGHT);

    const nextGround = groundTopAt(player.x);
    if (player.y >= nextGround) {
      player.y = nextGround;
      player.vy = 0;
    }

    player.animationTime += dt;
    player.attackTime += dt;
    player.auraTime += dt;
    player.beamTime += dt;

    if (player.attackTime < 0.35) {
      setAnimation("attack");
      return;
    }
    if (player.y < nextGround - 1) {
      setAnimation("jump");
      return;
    }
    if (Math.abs(player.vx) > 170) {
      setAnimation("run");
      return;
    }
    if (Math.abs(player.vx) > 18) {
      setAnimation("walk");
      return;
    }
    setAnimation("idle");
  }

  function drawBackground(now) {
    const sky = ctx.createLinearGradient(0, 0, 0, CANVAS_HEIGHT);
    sky.addColorStop(0, "#4d78a8");
    sky.addColorStop(0.5, "#173259");
    sky.addColorStop(1, "#09111c");
    ctx.fillStyle = sky;
    ctx.fillRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);

    backgroundNames.forEach((name, layerIndex) => {
      const sprite = sprites[name];
      const scale = 2.8;
      const width = sprite.w * scale;
      const height = sprite.h * scale;
      const y = 54 + layerIndex * 10;
      const speed = (layerIndex + 1) * 8;
      const offset = -((now * speed) % width);

      for (let x = offset - width; x < CANVAS_WIDTH + width; x += width - 8) {
        drawSprite(atlasImages, sprite, x, y, scale, false, 0.38 + layerIndex * 0.12);
      }
    });

    ctx.fillStyle = "rgba(255, 213, 120, 0.08)";
    ctx.fillRect(0, CANVAS_HEIGHT - 190, CANVAS_WIDTH, 180);
  }

  function drawPlatforms() {
    ctx.fillStyle = "#10192c";
    ctx.fillRect(0, FLOOR_Y + 48, CANVAS_WIDTH, CANVAS_HEIGHT - FLOOR_Y);

    platforms.forEach((platform) => {
      drawSprite(atlasImages, sprites[platform.sprite], platform.x, platform.y, 3);
    });
  }

  function drawEffects(nowSeconds, playerMetrics) {
    if (!playerMetrics) {
      return;
    }
    const playerAnchorX = playerMetrics.anchorX;
    const playerAnchorY = playerMetrics.anchorY;

    if (player.auraTime < 0.8) {
      const spriteName = frameAtTime(auraAnimation, nowSeconds * 1200 * 1.2);
      const sprite = sprites[spriteName];
      drawSprite(atlasImages, sprite, playerAnchorX, playerAnchorY - 92, 4, player.facing < 0, 0.86);
    }

    if (player.beamTime < 0.8) {
      const spriteName = frameAtTime(beamAnimation, nowSeconds * 1000 * 1.8);
      const sprite = sprites[spriteName];
      const beamX = playerAnchorX + player.facing * 132;
      drawSprite(atlasImages, sprite, beamX, playerAnchorY - 112, 3, player.facing < 0, 0.9);
    }
  }

  function drawPlayer() {
    const currentAnimation = animationByLabel(playerAnimations, player.animation, "idle");
    const spriteName =
      player.animation === "attack"
        ? frameAtTime(currentAnimation, player.attackTime * 1000, false)
        : frameAtTime(currentAnimation, player.animationTime * 1000, true);

    const sprite = sprites[spriteName];
    return drawSprite(atlasImages, sprite, player.x, player.y, PLAYER_SCALE, player.facing < 0);
  }

  function drawHud() {
    ctx.fillStyle = "rgba(6, 11, 24, 0.62)";
    ctx.fillRect(18, 18, 320, 78);

    ctx.fillStyle = "#f4f7ff";
    ctx.font = "18px Trebuchet MS";
    ctx.fillText(`Animation: ${player.animation}`, 34, 48);
    ctx.fillText(`Velocity: ${player.vx.toFixed(0)}, ${player.vy.toFixed(0)}`, 34, 74);

    ctx.fillStyle = "#a9bedb";
    ctx.font = "14px Trebuchet MS";
    ctx.fillText("Attack, aura, and beam all use extracted animation groups.", 34, 94);
  }

  function frame(now) {
    const dt = Math.min((now - lastTime) / 1000, 1 / 30);
    lastTime = now;

    update(dt);
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.globalAlpha = 1;
    ctx.clearRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);
    drawBackground(now / 1000);
    drawPlatforms();
    const playerMetrics = drawPlayer();
    drawEffects(now / 1000, playerMetrics);
    drawHud();

    requestAnimationFrame(frame);
  }

  statusNode.textContent = DEBUG_RENDER
    ? "Ready. Debug overlay enabled with ?debug."
    : "Ready. Use the keyboard controls below the canvas.";
  requestAnimationFrame(frame);
}

boot().catch((error) => {
  console.error(error);
  statusNode.textContent = "Failed to load demo assets. Serve the repo over HTTP.";
});
