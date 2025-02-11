import * as fs from 'fs';

export interface LLMResult {
    evalId: string;
    results: {
        prompts: {
            raw: string;
            id: string;
            provider: string;
        }[];
        results: {
            prompt: {
                raw: string;
            };
            promptId: string;
            provider: {
                id: string;
            };
            response: {
                output: string;
            };
        }[];
    };
}

export function parseLLMResults(filePath: string) {
    const fileContent = fs.readFileSync(filePath, 'utf-8');
    const jsonData: LLMResult = JSON.parse(fileContent);
    const extractedData = jsonData.results.results.map(result => {
        let promptText: {role: string, content: string}[] | string = result.prompt.raw
        try{
            let parseResult: {role: string, content: string}[] = JSON.parse(result.prompt.raw)
            promptText = parseResult
        } catch{
            ;
        }
        return {
        prompt: promptText,
        model: result.provider.id,
        response: result.response.output
        }
    }
);
    
    console.log(extractedData);
    return extractedData;
}

const filePath = '../../sample_outputs/sample_output_conversation.json'
parseLLMResults(filePath);
