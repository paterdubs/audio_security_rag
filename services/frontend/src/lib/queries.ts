/** Server state — TanStack Query. Không sao chép dữ liệu server vào store client. */

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';

import { apiGet, apiPostForm, apiPostJson } from './api';
import type { EventDetail, EventSummary, LocationOut, RagAnswer, Taxonomy, UploadResult } from './types';

export interface EventFilters {
  severity?: string;
  location?: string;
  class?: string;
}

export const queryKeys = {
  events: (filters: EventFilters) => ['events', filters] as const,
  event: (id: string) => ['event', id] as const,
  taxonomy: ['taxonomy'] as const,
  locations: ['locations'] as const,
};

function toQueryString(filters: EventFilters): string {
  const params = new URLSearchParams({ limit: '50' });
  for (const [key, value] of Object.entries(filters)) {
    if (value) params.set(key, value);
  }
  return params.toString();
}

export function useEvents(filters: EventFilters) {
  return useQuery({
    queryKey: queryKeys.events(filters),
    queryFn: () => apiGet<EventSummary[]>(`/events?${toQueryString(filters)}`),
  });
}

export function useEvent(eventId: string) {
  return useQuery({
    queryKey: queryKeys.event(eventId),
    queryFn: () => apiGet<EventDetail>(`/events/${eventId}`),
  });
}

export function useTaxonomy() {
  return useQuery({
    queryKey: queryKeys.taxonomy,
    queryFn: () => apiGet<Taxonomy>('/taxonomy'),
    // Taxonomy chỉ đổi khi người bảo trì sửa ontology_map.yaml rồi dựng lại service —
    // không có lý do gì hỏi lại trong một phiên làm việc.
    staleTime: Infinity,
  });
}

export function useLocations() {
  return useQuery({
    queryKey: queryKeys.locations,
    queryFn: () => apiGet<LocationOut[]>('/locations'),
    staleTime: Infinity,
  });
}

export function useUploadAudio() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: ({ file, locationId }: { file: File; locationId: string }) => {
      const form = new FormData();
      form.append('file', file);
      form.append('location_id', locationId);
      return apiPostForm<UploadResult>('/audio/upload', form);
    },
    onSuccess: () => {
      void client.invalidateQueries({ queryKey: ['events'] });
    },
  });
}

export function useRagQuery() {
  return useMutation({
    mutationFn: (question: string) => apiPostJson<RagAnswer>('/rag/query', { question }),
  });
}
