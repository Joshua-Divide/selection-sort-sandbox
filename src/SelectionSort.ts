export type StepType = 
    | 'startPass'
    | 'compare'
    | 'updateMin'
    | 'noUpdate'
    | 'swap'
    | 'noSwap'
    | 'finalize'
    | 'complete';

export interface SortStep {
    type: StepType;
    array: { value: number; id: string }[];
    i: number;
    j: number;
    minIndex: number;
    description: string;
    comparisons: number;
    swaps: number;
}

export const generateSelectionSortSteps = (
    input: { value: number; id: string }[]
): SortStep[] => {
    const steps: SortStep[] = [];
    const arr = [...input.map(item => ({ ...item }))];
    const n = arr.length;
    let comparisons = 0;
    let swaps = 0;

    const pushStep = (type: StepType, i: number, j: number, minIndex: number, desc: string) => {
        steps.push({
            type,
            array: arr.map(item => ({ ...item })),
            i, j, minIndex,
            description: desc,
            comparisons, swaps
        });
    };

    if (n === 0) {
        pushStep('complete', -1, -1, -1, 'Array is empty.');
        return steps;
    }

    pushStep('startPass', 0, 0, 0, 'Starting Selection Sort.');

    for (let i = 0; i < n; i++) {
        let minIndex = i;
        pushStep('startPass', i, i, minIndex, `Pass ${i + 1}: Finding the minimum element in the unsorted region [${i}..${n - 1}]. Initial min is at index ${i} (value ${arr[i].value}).`);

        for (let j = i + 1; j < n; j++) {
            comparisons++;
            pushStep('compare', i, j, minIndex, `Scanning: Comparing arr[${j}] (${arr[j].value}) with current minimum arr[${minIndex}] (${arr[minIndex].value}).`);

            if (arr[j].value < arr[minIndex].value) {
                minIndex = j;
                pushStep('updateMin', i, j, minIndex, `Found new minimum: ${arr[j].value} at index ${j}.`);
            } else {
                pushStep('noUpdate', i, j, minIndex, `${arr[j].value} is not smaller than ${arr[minIndex].value}. Keeping current minimum.`);
            }
        }

        if (minIndex !== i) {
            swaps++;
            [arr[i], arr[minIndex]] = [arr[minIndex], arr[i]];
            pushStep('swap', i, -1, minIndex, `Pass complete: Swapping start of unsorted region arr[${i}] with minimum found arr[${minIndex}].`);
        } else {
            pushStep('noSwap', i, -1, minIndex, `Pass complete: The element at arr[${i}] is already the minimum. No swap needed.`);
        }

        pushStep('finalize', i, -1, -1, `Element at index ${i} is now sorted and finalized.`);
    }

    pushStep('complete', n, -1, -1, 'Array is completely sorted.');
    return steps;
};
