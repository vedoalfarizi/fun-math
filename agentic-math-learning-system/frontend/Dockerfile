# frontend/Dockerfile
# ─────────────────────────────────────────────────────────────────────
# Node 20-alpine base keeps the image small while providing a modern
# Node.js runtime for the Vite development server.
# ─────────────────────────────────────────────────────────────────────
FROM node:20-alpine

WORKDIR /app

# Copy package.json first to leverage Docker layer caching —
# npm install only re-runs when dependencies change.
COPY package.json ./

RUN npm install

# Copy the rest of the source after installing dependencies.
COPY . .

# Expose port 3000 (Vite dev server default).
EXPOSE 3000

# --host makes Vite listen on 0.0.0.0 so the container port is reachable
# from the Docker Compose network and the host machine.
CMD ["npm", "run", "dev", "--", "--host"]
