import { useEffect, useCallback } from 'react';
import { useQueryClient, InvalidateQueryFilters } from '@tanstack/react-query';

export const useProfessorUpdates = (professorId: string) => {
    const queryClient = useQueryClient();

    const handleUpdate = useCallback(() => {
        // Invalidate and refetch relevant queries when updates occur
        const queryFilter: InvalidateQueryFilters = {
            predicate: (query) => 
                query.queryKey[0] === 'professor' ||
                query.queryKey[0] === 'recommendations' ||
                query.queryKey[0] === 'sentiment' ||
                query.queryKey[0] === 'metrics'
        };
        queryClient.invalidateQueries(queryFilter);
    }, [queryClient]);

    useEffect(() => {
        const ws = new WebSocket(`ws://${window.location.host}/ws/professors/${professorId}/`);

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            if (data.type === 'professor.update' && data.professor_id === professorId) {
                handleUpdate();
            }
        };

        return () => {
            ws.close();
        };
    }, [professorId, handleUpdate]);
};