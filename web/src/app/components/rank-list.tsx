"use client";

import { useEffect } from 'react';

export default function RankList() {
    useEffect(() => {
        const fetchData = async () => {
            const res = await fetch("/api/list");
            const data = await res.json();
            console.log(data);
        };
        fetchData();
    }, []);

    return (
        <div>
            test
        </div>
    );
}