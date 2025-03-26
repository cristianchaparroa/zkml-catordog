# 🐶🐱 zkml CatOrDog project

This project is a full-stack example for building decentralized applications (dapps). This app is a simple image classification example to classify images as either a cat or a dog. The app UI makes 2 calls to the backend:

- **zkml_backend_url/images**: The app UI sends the image to the zkml backend to be analyzed. The zkml backend uses the zkml inference engine to classify the image and returns the classification result to the app UI and to generate the respective ezkl proof to certify the result.
- **zkml_verifier_url/verifies**: The app UI sends a request to the zkml verifier to verify the classification proof.

## Quickstart

To get started with this project, follow the steps below:

1. Make sure you're already running the zkml servers in your localhost.

2. Install this folder project dependencies.

```bash
yarn install
```

3. Rename /packages/nextjs/env.example to .env (just rename it)

4. On a terminal, start your NextJS app:

```bash
yarn start
```

5. Visit your app on: `http://localhost:3000`. And that's it, you're running our dApp.

## CLI Usage

Depending on your package manager, substitute the work COMMAND with the appropiate one from the list.

$ yarn COMMAND
$ npm run COMMAND

Commands:

### CLI Frontend

| Command     | Description                                  |
| ----------- | -------------------------------------------- |
| start       | Starts the frontend server                   |
| test:nextjs | Runs the nextjs tests                        |
| vercel      | Deploys app to vercel                        |
| vercel:yolo | Force deploy app to vercel (ignoring errors) |
