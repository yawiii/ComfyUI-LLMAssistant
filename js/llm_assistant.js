import { app } from "/scripts/app.js";

app.registerExtension({
    name: "ComfyUI-LLMAssistant",
    async setup() {
        console.log("Loading ComfyUI-LLMAssistant extension...");
    },
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "LLMPromptAssistant") {
            console.log("Registering LLMPromptAssistant node...");
        }
    }
});
