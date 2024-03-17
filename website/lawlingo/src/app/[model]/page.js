'use client';
import { useState } from "react"
import { useParams } from 'react-router-dom';

export default function Model() {
    const [inputText, setInputText] = useState("");
    const { params } = useParams();

    const Summarise = () => {

    }

    return (
        <main className="flex min-h-screen flex-col items-center justify-between p-24">
            <div>
                <p className="text-bold text-xl">
                    {params}
                </p>
            </div>
            <form className="w-full">
                <div className="w-full mb-4 border border-gray-200 rounded-lg bg-gray-50 dark:bg-gray-700 dark:border-gray-600">
                    <div className="px-4 py-2 bg-white rounded-t-lg dark:bg-gray-800">
                        <label className="sr-only">Your legal text</label>
                        <textarea
                            rows="4"
                            className="w-full px-0 text-sm text-gray-900 bg-white border-0 dark:bg-gray-800 focus:ring-0 dark:text-white dark:placeholder-gray-400"
                            placeholder="Enter legal text..."
                            value={inputText}
                            onChange={() => setInputText(e.target.value)}
                            required
                        />
                    </div>
                    <div class="flex items-center justify-between px-3 py-2 border-t dark:border-gray-600">
                        <button
                            onClick={Summarise}
                            type="button"
                            className="text-gray-900 bg-white border border-gray-300 focus:outline-none hover:bg-gray-100 focus:ring-4 focus:ring-gray-100 font-medium rounded-lg text-sm px-5 py-2.5 me-2 mb-2 dark:bg-gray-800 dark:text-white dark:border-gray-600 dark:hover:bg-gray-700 dark:hover:border-gray-600 dark:focus:ring-gray-700">
                            Get Summary
                        </button>
                    </div>
                </div>
            </form>
        </main>
    )
}